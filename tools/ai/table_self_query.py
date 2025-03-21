import ast
import csv
import json
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Tuple
from typing import List, Optional, Dict, Hashable, Literal
from typing import Union

import chardet
import numpy as np
import pandas as pd
import pingouin
import scipy.stats
import statsmodels.api as sm
from dify_plugin.core.runtime import Session
from dify_plugin.entities.model.llm import LLMModelConfig
from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage
from loguru import logger
from pydantic import BaseModel, Field, field_validator

AI_DIR = Path(__file__).parent


def get_prompt_template(
        site: Literal["table_filter", "naming_master", "fdq", "classify", "table_interpreter", "answer"]
):
    path_system_prompt_table_filter = "templates/system_prompt_table_filter.xml"
    path_system_prompt_naming_master = "templates/system_prompt_naming_master.xml"
    path_system_prompt_fdq = "templates/system_prompt_fdq.xml"
    path_system_prompt_question_classifier = "templates/system_prompt_question_classifier.xml"
    path_system_prompt_table_interpreter = "templates/system_prompt_table_interpreter.xml"
    path_system_prompt_answer = "templates/system_prompt_answer.xml"

    site2path = {
        "table_filter": path_system_prompt_table_filter,
        "naming_master": path_system_prompt_naming_master,
        "fdq": path_system_prompt_fdq,
        "classify": path_system_prompt_question_classifier,
        "table_interpreter": path_system_prompt_table_interpreter,
        "answer": path_system_prompt_answer,
    }

    if site_path := site2path.get(site):
        system_prompt = AI_DIR.joinpath(site_path).read_text(encoding="utf8")
        return system_prompt

    return ""


PREVIEW_CODE_WRAPPER = """
<segment>
<question>{question}</question>
<question_type>{question_type}</question_type>
<code>
{query_code}
</code>
<output filename="{recommend_filename}">
{result_markdown}
</output>
</segment>
"""


class QueryStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class MetaData(BaseModel):
    row_count: int = Field(description="Number of results rows")
    columns: List[str] = Field(description="List of column names")
    dtypes: Dict[Hashable, str] = Field(description="Column data type")

    class Config:
        json_encoders = {np.dtype: str}


class QueryResult(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")
    query: str = Field(description="Original query")
    query_type: Optional[str] = Field(default="", description="Query Type")
    query_code: Optional[str] = Field(default="", description="The generated code for manipulating the table")
    recommend_filename: Optional[str] = Field(default="", description="Recommended file name")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Query result data")
    metadata: MetaData = Field(description="Result metadata")
    error: Optional[str] = Field(default=None, description="Error message")

    class Config:
        json_encoders = {
            np.integer: lambda x: int(x),
            np.floating: float,
            np.ndarray: lambda x: x.tolist(),
            pd.Timestamp: lambda x: x.isoformat(),
            datetime: lambda x: x.isoformat(),
        }

    @field_validator("data", mode="before")
    def convert_nan_to_none(cls, v):
        """Convert NaN value to None"""
        if isinstance(v, list):
            return [{k: None if pd.isna(v) else v for k, v in item.items()} for item in v]
        return v

    def get_recommend_filename(self, suffix: Optional[str] = None):
        """Generate file name suggestions based on query results"""
        name = self.recommend_filename or "output.xlsx"
        if isinstance(suffix, str) and suffix.startswith("."):
            name = f"{Path(name).stem}{suffix}"
        return name.strip()

    def to_llm_ready(self, *, storage_dir: Optional[Path] = None) -> str:
        wrapper_ = PREVIEW_CODE_WRAPPER.format(
            question=self.query,
            question_type=self.query_type,
            query_code=self.query_code,
            recommend_filename=self.get_recommend_filename(suffix=".md"),
            result_markdown=pd.DataFrame.from_records(self.data).to_markdown(index=False),
        ).strip()
        logger.success(f"transformed to LLM ready \n{wrapper_}")

        if isinstance(storage_dir, Path):
            storage_path = storage_dir / self.get_recommend_filename(suffix=".xml")
            storage_path.parent.mkdir(exist_ok=True, parents=True)
            storage_path.write_text(wrapper_, encoding="utf8")

        return wrapper_


class QueryOutputParser:

    @staticmethod
    def parse(
            df_result: Optional[pd.DataFrame],
            query: str,
            *,
            query_type: Optional[str] = "",
            query_code: Optional[str] = "",
            recommend_filename: Optional[str] = "",
            error: Optional[str] = "",
    ) -> QueryResult:
        """
        Convert DataFrame results to a standardized Pydantic model

        Args:
            query_code:
            df_result: Query the returned DataFrame
            query: Original query statement
            error: Error message (if any)
            recommend_filename: Recommended file names
            query_type:

        Returns:
            QueryResult: Standardized query result model
        """
        try:
            if isinstance(df_result, (pd.DataFrame, pd.Series)) and not df_result.empty:
                # 将DataFrame转换为字典列表
                data = df_result.to_dict("records")
                metadata = MetaData(
                    row_count=len(data),
                    columns=list(df_result.columns),
                    dtypes={col: str(dtype) for col, dtype in df_result.dtypes.items()},
                )
            elif df_result is not None:
                data = [{"output": f"{df_result}"}]
                metadata = MetaData(row_count=0, columns=[], dtypes={})
            else:
                data = []
                metadata = MetaData(row_count=0, columns=[], dtypes={})

            # 创建查询结果模型
            result = QueryResult(
                query=query,
                query_type=query_type,
                query_code=query_code,
                recommend_filename=recommend_filename,
                data=data,
                metadata=metadata,
                error=error,
            )

            return result

        except Exception as err:
            # 如果解析过程出错，返回错误结果
            error_result = QueryResult(
                query=query,
                query_type=query_type,
                query_code=query_code,
                recommend_filename=recommend_filename,
                data=[],
                metadata=MetaData(row_count=0, columns=[], dtypes={}),
                error=f"结果解析错误: {str(err)}",
            )
            return error_result


AVAILABLE_MODELS = Literal["qwen", "gemini", "coder"]


class TableLoader:
    def __init__(self):
        self.df = None
        self.schema_info = {}
        self.sample_data = []

    @staticmethod
    def _detect_file_encoding(file_path: Path) -> str:
        """Detect file encoding"""
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            return result["encoding"]

    @staticmethod
    def _find_valid_table_range(df: pd.DataFrame) -> Tuple[int, int]:
        """Find the start and end row index of a valid table"""

        # Count the number of non-null values per row
        row_counts = df.notna().sum(axis=1)
        max_count = row_counts.max()

        # Find the row range with stable number of data columns
        # 20% tolerance allowed
        valid_rows = row_counts[row_counts >= max_count * 0.8]
        start_idx = valid_rows.index[0]
        end_idx = valid_rows.index[-1]

        return start_idx, end_idx

    def _process_excel(self, file_path: Path) -> pd.DataFrame:
        """Process Excel files"""
        # First read all data without specifying the header
        return pd.read_excel(file_path, header=None)

    def _process_csv(self, file_path: Path) -> pd.DataFrame:
        """Processing CSV files"""
        encoding = self._detect_file_encoding(file_path)

        # First read all rows using csv.reader
        with open(file_path, "r", encoding=encoding) as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # Convert to DataFrame for processing
        raw_df = pd.DataFrame(rows)
        start_idx, end_idx = self._find_valid_table_range(raw_df)

        # Extract valid data range
        valid_df = raw_df.iloc[start_idx: end_idx + 1].reset_index(drop=True)
        return valid_df

    def format_stock_codes(self, keywords: List[str] = None) -> None:
        """
        Scan the column name of the DataFrame and format the columns containing keywords and of type int:
        1. Convert to str type
        2. Complement zero to 6 digits for non-null values

        Args:
            keywords:

        Returns:

        """
        if keywords is None:
            keywords = ["证券代码", "code_id", "symbol"]

        # 获取所有列名
        columns = self.df.columns.tolist()

        # 遍历列名
        for col in columns:
            # 检查列名是否包含关键词
            if any(keyword.lower() in col.lower() for keyword in keywords):
                # 检查数据类型是否为int
                if pd.api.types.is_integer_dtype(self.df[col]):
                    # 转换为字符串类型
                    self.df[col] = self.df[col].astype(str)

                    # 对非空值补零到6位
                    self.df[col] = self.df[col].apply(lambda x: x.zfill(6) if pd.notna(x) else x)

                    logger.debug(f"列 '{col}' 已格式化为6位证券代码")

        return None

    @staticmethod
    def _detect_header_row(df: pd.DataFrame) -> Union[int, None]:
        """
        Automatically detect the row where the table header is located. Returns the row index (starting from 0), and None if not detected.

        Improved detection rules:
        1. Basic conditions: The number of non-null values in the row is sufficient (>50%), and the value is basically unique
        2. Data type characteristics: The header is usually a string type
        3. Length characteristics: The length of the header string is usually moderate, not too long or too short
        4. Numerical ratio: The header row usually does not contain a large number of numerical values
        5. Repeat value: The table header row should have fewer duplicate values
        """
        max_check_rows = min(15, len(df))
        best_score = -1
        best_row = None

        for idx in range(max_check_rows):
            row = df.iloc[idx]
            score = 0

            # 1. 检查非空值比例和唯一值
            num_non_na = row.notna().sum()
            num_unique = row.nunique()
            non_na_ratio = num_non_na / len(df.columns)

            if non_na_ratio < 0.5:  # 非空值比例过低
                continue

            # 基础分数：非空值比例 * 唯一值比例
            score += non_na_ratio * (num_unique / num_non_na)

            # 2. 检查数据类型
            str_count = sum(1 for x in row if isinstance(x, str))
            str_ratio = str_count / num_non_na
            score += str_ratio * 2  # 字符串比例权重加倍

            # 3. 检查字符串长度
            if str_count > 0:
                str_lengths = [len(str(x)) for x in row if isinstance(x, str)]
                avg_len = sum(str_lengths) / len(str_lengths)
                # 理想的表头长度在2-20之间
                if 2 <= avg_len <= 20:
                    score += 1
                elif avg_len > 50:  # 可能是数据行
                    score -= 1

            # 4. 检查数值比例
            num_count = sum(
                1 for x in row if isinstance(x, (int, float)) and not isinstance(x, bool)
            )
            num_ratio = num_count / num_non_na
            if num_ratio > 0.5:  # 数值比例过高
                score -= 1

            # 5. 检查重复值
            duplicate_ratio = 1 - (num_unique / num_non_na)
            if duplicate_ratio > 0.2:  # 重复值比例过高
                score -= 1

            # 6. 检查特殊标记（可选）
            special_keywords = {
                "序号",
                "编号",
                "名称",
                "代码",
                "日期",
                "id",
                "name",
                "code",
                "date",
            }
            keyword_matches = sum(
                1
                for x in row
                if isinstance(x, str) and any(k in str(x).lower() for k in special_keywords)
            )
            if keyword_matches > 0:
                score += 0.5

            # 更新最佳分数
            if score > best_score:
                best_score = score
                best_row = idx

        # 要求最小分数阈值
        return best_row if best_score >= 1.5 else None

    def _extract_schema_and_samples(self) -> None:
        """Extract table schema information and sample data"""

        def _convert_for_json(obj):
            """Convert an object to make it JSON serializable"""
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return str(obj)

        # 获取schema信息
        self.schema_info = {
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "shape": self.df.shape,
        }

        # 获取随机样例(最多b行)
        sample_size = min(5, len(self.df))
        samples = self.df.sample(n=sample_size).to_dict("records")
        self.sample_data = [
            {k: _convert_for_json(v) for k, v in record.items()} for record in samples
        ]

    def load_table(self, file_path: Union[str, Path]) -> None:
        """Load the table file, process the head and tail comments, and automatically detect the head rows of the table"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            valid_df = self._process_csv(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            valid_df = pd.read_excel(file_path, header=None)
        else:
            raise ValueError("Unsupported file formats. Please use table file")

        # 检测表头行
        header_row = self._detect_header_row(valid_df)
        if header_row is not None:
            self.df = valid_df.iloc[header_row:].reset_index(drop=True)
            self.df.columns = valid_df.iloc[header_row]
        else:
            # 默认使用第一行作为表头
            self.df = valid_df.iloc[1:].reset_index(drop=True)
            self.df.columns = valid_df.iloc[0]

        self.format_stock_codes()
        self._extract_schema_and_samples()


class TableQueryEngine:
    def __init__(self, session: Session, dify_model_config: LLMModelConfig):
        self.session = session
        self.dify_model_config = dify_model_config

        self.df = None
        self.schema_info = {}
        self.sample_data = []

    def _invoke_dify_llm(
            self,
            user_content: str,
            system_prompt: str = None,
            temperature: float = 0,
            max_tokens: int = 4096,
    ) -> str:
        model_config = self.dify_model_config.model_dump().copy()
        model_config["completion_params"] = {"max_tokens": max_tokens, "temperature": temperature}
        llm_result = self.session.model.llm.invoke(
            model_config=model_config,
            prompt_messages=[
                SystemPromptMessage(content=system_prompt),
                UserPromptMessage(content=user_content),
            ],
            stream=False,
        )

        return llm_result.message.content

    def load_table(self, file_path: Union[str, Path]) -> None:
        tl = TableLoader()
        tl.load_table(file_path)

        self.df = tl.df
        self.schema_info = tl.schema_info
        self.sample_data = tl.sample_data

    def second_level_classify(self, natural_query: str):
        schemas = {
            "columns": self.schema_info["columns"],
            "dtypes_dict": self.schema_info["dtypes"],
            "shape": self.schema_info["shape"],
            "sample_data": json.dumps(self.sample_data[0], indent=2, ensure_ascii=False),
        }
        system_prompt = f"""
<task>
You are a question classifier, determining whether a user query is a <basic data query> task. Output "YES" if it is.
</task>
<COT>
Judgment: 1. Does the query explicitly request a query? 2. Does the query request a listing of data?
For example, <not belonging> case: The table is an exam grade sheet, and the query requires calculating the median score of students in a certain region (i.e., data not explicitly given in the grade sheet).
For example, <belonging> case: The table is an exam grade sheet, and the query is to query the scores of a subject or the scores of students in a certain region.
</COT>
<schemas>
Below are the headers, sample data, shape, and description information of the knowledge base tables.
{schemas}
</schemas>
<limitations>
You can only output "YES" or "NO" and cannot output any other information.
</limitations>
        """.strip()
        runtime_params = {
            "system_prompt": system_prompt,
            "user_content": natural_query,
            "temperature": 0,
            "max_tokens": 512,
        }
        return self._invoke_dify_llm(**runtime_params)

    def _generate_query_code(
            self, natural_query: str, agent: Literal["table_filter", "table_interpreter"]
    ) -> str:
        """Generate query code using LLM

        Args:
            natural_query:

        Returns:
            str:
        """
        system_prompt = get_prompt_template(agent).format(
            columns_list=self.schema_info["columns"],
            dtypes_dict=self.schema_info["dtypes"],
            shape_tuple=self.schema_info["shape"],
            sample_data=json.dumps(self.sample_data, indent=2, ensure_ascii=False),
        )
        user_content = f"<query>{natural_query}<query>"

        runtime_params = {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "temperature": 0,
            "max_tokens": 4096,
        }

        code = self._invoke_dify_llm(**runtime_params)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()

        return code

    def _generate_filename(self, natural_query: str, code: str) -> str:
        system_prompt = get_prompt_template("naming_master")

        user_content = f"""
<natural_query>
{natural_query}
</natural_query>
<code>
```python
{code}
```
</code>
        """.strip()
        runtime_params = {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "temperature": 0.3,
            "max_tokens": 512,
        }

        return self._invoke_dify_llm(**runtime_params)

    def _safe_execute_code(self, code: str) -> Any:
        """
        Safely execute dynamically generated code

        Args:
            code: Python code to execute

        Returns:
        """
        try:
            # Parse the code to ensure the syntax is correct
            ast.parse(code)

            # Create an isolated namespace
            namespace = {
                # Basic library
                "pd": pd,
                "np": np,
                "df": self.df,
                # Statistical related
                "sm": sm,
                "stats": scipy.stats,
                "pingouin": pingouin,
            }

            # Execute function definitions using exec
            exec(code, namespace)

            # Call the generated function
            if "execute_query" not in namespace:
                raise ValueError("The executed_query function was not found in the generated code")

            result = namespace["execute_query"](self.df)
            return result

        except Exception as e:
            error_context = {
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }

            logger.error("Code execution failed", extra={"error_context": error_context})

            return str(traceback.format_exc())

    def query(
            self, natural_query: str, enable_classifier: bool = True, retry_times: int = 3
    ) -> QueryResult:
        if self.df is None:
            return QueryOutputParser.parse(None, natural_query, error="Tabular data not loaded")

        try:

            # ====================
            # Task Classifier
            # ====================
            # Post-Judgement - Problem Scenario Classification, and then use different branches to generate code
            if enable_classifier:
                # query_type = self.get_query_classification(natural_query)
                is_search = self.second_level_classify(natural_query)
                query_type = "DataQuery" if "YES" in is_search else "DataAnalysis"
            # Ignore judgment - To the query is the `Basic Data Query` class,
            # guide subsequent workflows to print the results instead of re-investment into memory
            else:
                query_type = "DataQuery"

            # ====================
            # Task Execution
            # ====================
            query_code = ""
            df_result = None
            for _ in range(retry_times):
                # Generate code
                query_code = self._generate_query_code(
                    natural_query=natural_query,
                    agent="table_filter" if query_type == "DataQuery" else "table_interpreter",
                )

                # logger.debug(f"[{query_type}]Generate code: \n{query_code}")
                df_result = self._safe_execute_code(query_code)
                # What is returned is not an error message
                if not isinstance(df_result, str):
                    break

                # TODO: debugger strategy

            # ========================================
            # Generate recommended file names
            # ========================================
            recommend_filename = self._generate_filename(natural_query, query_code)

            # ========================================
            # Parses and returns the result
            # ========================================
            result = QueryOutputParser.parse(
                df_result,
                natural_query,
                query_type=query_type,
                query_code=query_code,
                recommend_filename=recommend_filename,
            )
            return result

        except Exception as err:
            return QueryOutputParser.parse(pd.DataFrame(), natural_query, error=str(err))
