import json
from typing import List, Literal

import requests
from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(".env")
import os

host = os.environ["host"]
port = os.environ["port"]
temp_files_path = "./temp"


class ContentExtractor(object):
    def __init__(self, temp_dir=temp_files_path, k=3):
        self.k = k
        self.temp_dir = temp_dir
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.docs = [Document("None", metadata={"path": "None"})]
        self.vectorstore = Chroma.from_documents(
            documents=self.docs,
            collection_name="temp_data",
            embedding=embeddings,
        )
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = self.k
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(search_kwargs={"k": self.k}),
                self.bm25_retriever,
            ],
            weights=[0.4, 0.6],
        )

    def update_new_docs(self, files):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=250,
            separators=["\n\n\n\n", "\n\n\n", "\n\n", "\n", ""],
        )
        new_docs = []
        for file in files:
            with open(file, "r") as f:
                texts = f.read()
            name = file.split("/")[-1].replace(".txt", "")
            new_docs.append(Document(texts, metadata={"path": name}))
        new_docs = text_splitter.split_documents(new_docs)
        return new_docs

    def update(self, files):
        new_docs = self.update_new_docs(files)
        self.vectorstore.add_documents(new_docs)
        self.docs.extend(new_docs)

        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = self.k
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.vectorstore.as_retriever(search_kwargs={"k": self.k}),
                self.bm25_retriever,
            ],
            weights=[0.4, 0.6],
        )

    def query(self, q):
        info = []
        results = self.hybrid_retriever.get_relevant_documents(q)
        for res in results:
            if res not in info:
                info.append(res)
        return info


content_extractor = ContentExtractor()


@tool
def step_formatter(steps: List[str]):
    """Summary
    List the question solving steps by increasing order
    Args:
        steps:List[str]: the question solving steps in increasing order
    """
    return {"steps": steps}


@tool
def question_formatter(thought: str, questions: List[str]):
    """Summary
    This function performs a ReAct search using search engine and LLM, which is designed
    to provide comprehensive, accurate, and trusted results.
    It's particularly useful for answering questions about current events.
    Take thoughts and a list of queries convert them into Queries object
    Args:
        thought:str : the thought of these queries
        questions:List[str]: the queries in a list
    """
    return {"search_queries": questions}


@tool
def queries_formatter(thought: str, queries: List[str]):
    """Summary
    This function performs a search using search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    Take thoughts and a list of queries convert them into Queries object
    Args:
        thought:str : the thought of these queries
        queries:List[str]: the queries in a list
    """
    return {"search_queries": queries}


@tool
def validate_formatter(done: bool, new_plan: List[str]):
    """Summary
    if the information of the topic is enough (do not need to regenerate step by step plan)
    done = True
    new_plan = []
    else
    done = False
    new_plan = [step1 plan, step2 plan, ......]

    Args:
        done (bool): True: need to re-generate step by step plan, False: the information completed enough can generate report
        new_plan (List[str]):  new step by step plan

    """


@tool
def feedback_formatter(grade: Literal["pass", "fail"], follow_up_queries: List[str]):
    """Summary
    Take grad and follow up queries convert them into Feedback object
    Args:
        grade (Literal[pass,fail]): Evaluation result indicating whether the response meets requirements ('pass') or needs revision ('fail').
        follow_up_squeries (List[str]): List of follow-up search queries.
    """
    ...


@tool
def section_formatter(name: str, description: str, research: bool, content: str):
    """Summary
    Take name, description, research, content and convert them into Section object
    Args:
        name (str): Name for this section of the report.
        description (str): Brief overview of the main topics and concepts to be covered in this section.
        research (bool): Whether to perform web research for this section of the report.
        content (str): The content of the section.
    """
    return {
        "name": name,
        "description": description,
        "research": research,
        "content": content,
    }


@tool
def run_python(repr: str):
    """
    This tool is a tool exectuting python code for scenarios requiring logical operations, complex calculations, data interaction, and analysis, including:
    For any estimations regarding numerical data, such as target price estimations or simulations of different parameters under specific circumstances, be sure to call this tool.

    - Return simple String output
    - Simulation (模擬)
    - Numerical Calculation (數值運算), such as statistical analysis, linear algebra, etc.
    - Numerical Prediction (數值預測), such as time series forecasting, regression analysis, etc.
    - Value Estimation (價值估計), such as financial modeling, asset valuation, etc.
    - Plotting (繪圖)
    - Comparison (比較)
    - Data processing and manipulation (資料處理與操作)
    - Algorithmic problem solving (演算法問題解決)

    If you need to inspect the content or results of this code, please use
    the `print()` function within the Python code to output the values.
    Multiple printouts are possible, so it is recommended to precede each printed result with a variable name or a clear description of what the value represents for better understanding and debugging.

    Usage Demo:
        =========================================================================
        Case 0 Simple Return Strings
            repr='''# Statements1 #Statements2 .... '''
        =========================================================================
        Case 1 Dividend Discount Model, DDM:
            repr='''
            next_year_dividend = 5
            growth_rate = 0.03
            required_rate_of_return = 0.08

            if required_rate_of_return <= growth_rate:
                print("必要報酬率必須大於股息增長率才能使用此模型。")
            else:
                target_price = next_year_dividend / (required_rate_of_return - growth_rate)
                print(f"根據股息折現模型，該公司的目標股價估計為：${target_price:.2f}")
            '''
        =========================================================================
        Case 2 Discounted Cash Flow, DCF:
            repr='''
            fcf = [10000000, 12000000, 14000000, 16000000, 18000000]
            terminal_growth_rate = 0.02
            wacc = 0.10
            shares_outstanding = 10000000
            pv_fcf = 0
            for i, cash_flow in enumerate(fcf):
                pv_fcf += cash_flow / (1 + wacc)**(i + 1)

            terminal_value = fcf[-1] * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
            pv_terminal_value = terminal_value / (1 + wacc)**len(fcf)
            equity_value = pv_fcf + pv_terminal_value
            target_price_per_share = equity_value / shares_outstanding
            print(f"根據自由現金流量折現模型，該公司的每股目標股價估計為：${target_price_per_share:.2f}")
            '''

        =========================================================================
        Case 3 Price-to-Earnings Ratio, P/E
            repr='''
            peer_group_pe_ratio = 15
            expected_eps = 8
            target_price = peer_group_pe_ratio * expected_eps
            print(f"根據本益比倍數法，該公司的目標股價估計為：${target_price:.2f}")
            '''
        =========================================================================
        Case 4: New Solar Plant Project Revenue Monte Carlo Simulation
            repr='''
            import numpy as np
            import pandas as pd

            # Number of simulations
            num_simulations = 10000

            # Plant characteristics
            plant_capacity_mw = 50
            hours_in_year = 365 * 24
            ops_cost_inflation_year2 = 0.03

            # Parameter ranges
            capacity_factor_range = np.array([0.20, 0.30])
            electricity_price_range = np.array([40, 70])  # USD per MWh
            operational_costs_range = np.array([25000, 40000])  # USD per MW capacity per year
            rec_value_range = np.array([5, 15])  # USD per MWh

            # Store simulation results
            net_revenue_year1 = np.zeros(num_simulations)
            net_revenue_year2 = np.zeros(num_simulations)

            for i in range(num_simulations):
                # Random sampling for Year 1
                capacity_factor = np.random.uniform(capacity_factor_range[0], capacity_factor_range[1])
                electricity_price = np.random.uniform(electricity_price_range[0], electricity_price_range[1])
                annual_op_costs_per_mw_year1 = np.random.uniform(operational_costs_range[0], operational_costs_range[1])
                rec_value = np.random.uniform(rec_value_range[0], rec_value_range[1])

                # Calculate Year 1 metrics
                annual_energy_production_mwh = plant_capacity_mw * capacity_factor * hours_in_year

                electricity_sales_revenue_year1 = annual_energy_production_mwh * electricity_price
                rec_revenue_year1 = annual_energy_production_mwh * rec_value
                total_gross_revenue_year1 = electricity_sales_revenue_year1 + rec_revenue_year1

                total_operational_costs_year1 = annual_op_costs_per_mw_year1 * plant_capacity_mw
                net_revenue_year1[i] = total_gross_revenue_year1 - total_operational_costs_year1

                # Calculate Year 2 metrics
                # Assume capacity factor, electricity price, and REC value ranges are the same for simplicity
                # Operational costs inflate
                annual_op_costs_per_mw_year2 = annual_op_costs_per_mw_year1 * (1 + ops_cost_inflation_year2)

                # Re-sample for variables that might change or have persistent but uncertain values
                # For this example, we assume new independent samples for price and REC for year 2,
                # and capacity factor can also be re-sampled if it's considered to vary year-to-year.
                # If these were meant to be correlated or trended, the model would be more complex.
                # Here, we'll assume they are independently sampled each year from the same distribution for simplicity as per prompt.
                capacity_factor_y2 = np.random.uniform(capacity_factor_range[0], capacity_factor_range[1])
                electricity_price_y2 = np.random.uniform(electricity_price_range[0], electricity_price_range[1])
                rec_value_y2 = np.random.uniform(rec_value_range[0], rec_value_range[1])

                annual_energy_production_mwh_y2 = plant_capacity_mw * capacity_factor_y2 * hours_in_year

                electricity_sales_revenue_year2 = annual_energy_production_mwh_y2 * electricity_price_y2
                rec_revenue_year2 = annual_energy_production_mwh_y2 * rec_value_y2
                total_gross_revenue_year2 = electricity_sales_revenue_year2 + rec_revenue_year2

                total_operational_costs_year2 = annual_op_costs_per_mw_year2 * plant_capacity_mw
                net_revenue_year2[i] = total_gross_revenue_year2 - total_operational_costs_year2

            # Output statistical summary
            results = pd.DataFrame({'Net_Operating_Revenue_Year1_USD': net_revenue_year1,
                                    'Net_Operating_Revenue_Year2_USD': net_revenue_year2})
            print(f"Evergreen Energy Solutions (EES.US) Project Sunstone ({plant_capacity_mw} MW) Net Operating Revenue Forecast (USD)")
            print(results.describe(percentiles=[0.05, 0.5, 0.95]))
            '''
        =========================================================================
        Case 5 Time series forcasting
            repr='''
            import pandas as pd
            import numpy as np
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.arima.model import ARIMA
            import warnings

            # Suppress warnings for a cleaner output (optional)
            warnings.filterwarnings("ignore")

            # Generate synthetic historical sales data (36 months)
            np.random.seed(42) # for reproducibility
            base_sales = 1000
            trend_factor = 10 # units increase per month
            months = np.arange(1, 37)
            # Trend component
            sales_trend = base_sales + months * trend_factor
            # Seasonal component (12-month cycle)
            sales_seasonality = 150 * np.sin(2 * np.pi * months / 12) + 100 * np.cos(2 * np.pi * (months-3) / 12)
            # Noise component
            sales_noise = np.random.normal(0, 75, 36)
            # Combine components
            historical_sales = sales_trend + sales_seasonality + sales_noise
            historical_sales = np.maximum(historical_sales, 200) # Ensure sales are not unrealistically low or negative

            # Create a pandas Series with a DatetimeIndex (optional, but good practice)
            date_rng = pd.date_range(start='2022-01-01', end='2024-12-01', freq='MS')
            sales_series = pd.Series(historical_sales, index=date_rng)

            # SARIMA model parameters (example, can be optimized with ACF/PACF or auto_arima)
            # (p,d,q)
            order = (1, 1, 1)
            # (P,D,Q,s) where s is the seasonal period (12 for monthly data)
            seasonal_order = (1, 1, 1, 12)

            # Fit the SARIMA model
            try:
                model = SARIMAX(sales_series,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                                initialization='approximate_diffuse') # Added initialization
                results = model.fit(disp=False) # Set disp=False to reduce console output during fitting
            except Exception as e_sarimax:
                print(f"SARIMAX failed: {e_sarimax}. Trying ARIMA without seasonality as fallback.")
                try:
                    # Fallback to ARIMA if SARIMAX has issues (e.g., convergence in simple examples)
                    model = ARIMA(sales_series, order=(1,1,1)) # Simpler ARIMA
                    results = model.fit()
                except Exception as e_arima:
                    print(f"ARIMA fallback also failed: {e_arima}")
                    # If both fail, produce a naive forecast or error
                    results = None # Indicate model fitting failure

            if results:
                # Forecast for the next 6 months
                forecast_steps = 6
                forecast = results.get_forecast(steps=forecast_steps)
                predicted_sales = forecast.predicted_mean

                # Create a date range for the forecast
                forecast_index = pd.date_range(start=sales_series.index[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
                forecast_df = pd.DataFrame({'Date': forecast_index.strftime('%Y-%m-%d'), 'Forecasted_Sales_Units': predicted_sales.values.round(0)})

                print("Global Retail Corp (GRC.US) - Product X Monthly Sales Forecast (Next 6 Months):")
                print(forecast_df.to_string(index=False))
            else:
                print("Could not generate forecast due to model fitting issues.")

            '''
    Args:
        repr[str] : The Python code to be executed, provided as a string.

    Returns:
        str: The standard output from the execution of the Python code. This will include any output generated by the `print()` function within the executed code.
    """
    exec_tool = PythonREPLTool()
    return exec_tool(repr)


@tool
def selenium_api_search(search_queries: List[str]):
    """Summary
    This function performs a search using search engine, which is designed to provide comprehensive, accurate, and trusted results.
    Please convert question into search engine suitable queries to search queries.
    Take thoughts and a list of queries convert them into Queries object
    Args:
        thought:str : the thought of these queries
        queries:List[str]: the queries in a list
    """
    search_docs = []
    for query in search_queries:
        output = requests.get(
            f"http://{host}:{port}/search_and_crawl",
            params={
                "query": query,
                "include_raw_content": True,
                "max_results": 5,
                "timeout": 40,
            },
        )
        output = json.loads(output.content)
        large_files = []
        for result in output["results"]:
            result["title"] = result["title"].replace("/", "_")
            if result.get("raw_content", "") is None:
                continue
            try:
                if len(result.get("raw_content", "")) >= 5000:
                    file_path = f"{temp_files_path}/{result['title']}.txt"
                    with open(file_path, "w") as f:
                        f.write(result["raw_content"])
                    result["raw_content"] = ""
                    large_files.append(file_path)
            except Exception as e:
                print(e)
                print(result)

        if len(large_files) > 0:
            content_extractor.update(large_files)
            search_results = content_extractor.query(query)
            for results in search_results:
                output["results"].append(
                    {
                        "url": results.metadata["path"],
                        "title": results.metadata["path"],
                        "content": "Raw content part has the most relevant information.",
                        "raw_content": results.page_content,
                    }
                )

        search_docs.append(output)
    return search_docs
