import pandas as pd
from sqlalchemy import create_engine
import dateparser
import numpy as np

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 11)

# Credentials to connect to the database
username = "username"
password = "DB_password"
hostname = "DB_host"
dbname = "DB_name"


# Process the initial data
def first_data_handle():
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    # Read the data
    data = pd.read_sql_table("sales2", engine)
    # Round the prices so as to have two decimals
    data.product_price = round(data.product_price, 2)
    # Find the data with zero product_price
    zero_price_data = data.where(data.product_price == 0.0).dropna(axis=0)
    # Remove the zero price data
    data.drop(zero_price_data.index, axis=0, inplace=True)
    data.index = range(len(data))
    with pd.option_context('mode.use_inf_as_na', True):
        data = data.dropna(axis=0)
    data.index = range(len(data))

    # Write the processed data
    data.to_sql(name="sales", con=engine, if_exists="replace", index=False, chunksize=1000)


# Process the data and find the products that were purchased more than num_of_sales
def products_with_sales(num_of_sales=1000):
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    # Read the data
    data = pd.read_sql_table("sales", engine)
    data["order_timestamp"] = data["order_timestamp"].astype("str")

    # Find the products that were purchased more than 1000 times
    products_with_high_vol = data.groupby(["product_id"]).product_quantity.sum()
    products_with_high_vol = products_with_high_vol.where(products_with_high_vol >= num_of_sales).dropna(axis=0)
    data = data.set_index("product_id").join(products_with_high_vol, rsuffix="_total")
    data.dropna(axis=0, inplace=True)
    data.drop(columns="product_quantity_total", inplace=True)

    # Find in which week the products were purchased
    # compared to the date of the first order of the dataset
    data.sort_values(by="order_timestamp", inplace=True)
    first_date = dateparser.parse(data.iloc[0].order_timestamp)
    last_date = dateparser.parse(data.iloc[-1].order_timestamp)
    shift = 6 - ((last_date - first_date).days % 7)
    data["week"] = 0
    weeks = []
    for i in range(data.shape[0]):
        date = dateparser.parse(data.iloc[i].order_timestamp)
        week = (((date - first_date).days + shift) // 7) + 1
        weeks.append(week)
    data["week"] = weeks

    data["product_id"] = data.index
    data.index = range(len(data))
    with pd.option_context('mode.use_inf_as_na', True):
        data = data.dropna(axis=0)
    data.index = range(len(data))

    # Write the processed data
    data.to_sql(name=f"products_{num_of_sales}_sales", con=engine, index=False, if_exists="replace", chunksize=10)


# Data aggregation. Collect the data in a weekly basis
def create_week_data():
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    # Read the data
    data = pd.read_sql_table("products_1000_sales", engine)
    data["order_timestamp"] = data["order_timestamp"].astype("str")
    # Demand per week
    demand_data = data.groupby(["product_id", "week"]).product_quantity.sum()
    # Mean price per week
    price_data = data.groupby(["product_id", "week"]).product_price.mean().round(2)
    week_data = pd.concat([demand_data, price_data], axis=1)
    week_data = week_data.reset_index(level=["product_id", "week"])
    # Assume a cost for each product based on the minimum of the price
    products = week_data.product_id.unique()
    cost = pd.Series()
    max_prices = pd.Series()
    for product in products:
        min_price = week_data.loc[week_data.product_id == product].product_price.min()
        max_price = week_data.loc[week_data.product_id == product].product_price.max()
        temp_ind = week_data.loc[week_data.product_id == product].index
        for i in temp_ind:
            cost.loc[i] = round(0.8 * min_price, 2)
            max_prices.loc[i] = round(1.2 * max_price, 2)

    week_data["product_cost"] = cost
    week_data["product_max_bound"] = max_prices

    week_data.to_sql(name="week_data", con=engine, index=False, if_exists="replace", chunksize=1000)


def full_weeks(missing_weeks, total_weeks):
    '''
        Return a dataframe with full weeks of our dataset
        Take one product and fill the empty weeks with zeros
    '''
    full_weeks = pd.DataFrame(columns=missing_weeks.columns)
    for week in range(1, total_weeks+1):
        flag = True
        if not missing_weeks.loc[missing_weeks["week"] == week].week.values:
            flag = False
        temp = []
        temp.append(missing_weeks.loc[0]["product_id"])
        temp.append(week)
        if flag:
            temp.append(missing_weeks.loc[missing_weeks["week"] == week].product_quantity.values[0])
            temp.append(missing_weeks.loc[missing_weeks["week"] == week].product_price.values[0])
        else:
            temp.append(0)
            temp.append(0)
        temp.append(missing_weeks.loc[0]["product_cost"])
        temp.append(missing_weeks.loc[0]["product_max_bound"])
        full_weeks.loc[(week - 1)] = temp
    return full_weeks.copy()


def nn_row(row, full_weeks, number_of_weeks):
    '''
        Return a row in the desired format for the neural network
    '''
    row_data = []
    row_data.append(row["week"])
    row_data.append(row["product_cost"])
    row_data.append(row["product_max_bound"])
    row_data.append(row["product_id"])
    week = row["week"]
    weeks = np.ndarray(shape=(number_of_weeks, 2))
    for i in range(1, number_of_weeks+1):
        temp_week = week - i
        if temp_week < 1:
            p = 0
            q = 0
        else:
            p = full_weeks.loc[full_weeks["week"] == temp_week].product_price.values[0]
            q = full_weeks.loc[full_weeks["week"] == temp_week].product_quantity.values[0]
        weeks[(number_of_weeks - i), 0] = p
        weeks[(number_of_weeks - i), 1] = q

    for i in range(number_of_weeks):
        row_data.append(weeks[i, 0])
        row_data.append(weeks[i, 1])

    row_data.append(row["product_price"])
    row_data.append(row["product_quantity"])
    return row_data


# Process the data to be in the desired format for the neural network
def create_nn_data(number_of_weeks=16):
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    week_data = pd.read_sql_table("week_data", engine)
    total_weeks = week_data.week.max()

    columns = ["week", "product_cost", "product_max_bound", "product_id"]
    for i in range(1, number_of_weeks+2):
        columns.append("P{:d}".format(i))
        columns.append("Q{:d}".format(i))
    nn_data = pd.DataFrame(columns=columns)
    products = week_data.product_id.unique()
    for product in products:
        temp_product = week_data.loc[week_data["product_id"] == product]
        temp_product.index = range(temp_product.shape[0])
        full_week = full_weeks(temp_product.copy(), total_weeks)

        temp_data = pd.DataFrame(columns=columns)
        for index, row in temp_product.iterrows():
            temp_data.loc[index] = nn_row(row, full_week.copy(), number_of_weeks)
        nn_data = nn_data.append(temp_data, ignore_index=True)

    nn_data.to_sql(name="nn_data", con=engine, index=False, if_exists="replace", chunksize=1000)


# Process the data to be in the desired format for the particle swarm optimization
def pso_data():
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    nn_data = pd.read_sql_table("nn_data", engine)
    total_weeks = nn_data.week.max()
    number_of_weeks = int((nn_data.shape[1] - 6) / 2)

    data = nn_data.loc[nn_data["week"] == total_weeks].copy()
    data.index = range(len(data))

    for i in range(1, number_of_weeks+1):
        data.loc[:, f"P{i}"] = data[f"P{i+1}"]
        data.loc[:, f"Q{i}"] = data[f"Q{i+1}"]
    data.drop(columns=["week", f"P{number_of_weeks+1}", f"Q{number_of_weeks+1}"], inplace=True)
    pso = pd.DataFrame(columns=data.columns)
    pso["product_min_bound"] = 0

    # Read the data to see which products will be priced dynamically
    opt_data = pd.read_sql_table("data_for_optimization", engine)
    products = opt_data["product_id"]
    for product in products:
        temp = data.loc[data["product_id"] == product].copy()
        arxiki = opt_data.loc[opt_data["product_id"] == product].arxikiTimi.values
        teliki = opt_data.loc[opt_data["product_id"] == product].telikiTimi.values
        percentage = 1 - (teliki / arxiki)
        max = (1 - (percentage - 0.1)) * arxiki
        if max > arxiki:
            max = arxiki
        min = (1 - (percentage + 0.1)) * arxiki
        temp.loc[:, "product_max_bound"] = round(float(max), 2)
        temp["product_min_bound"] = round(float(min), 2)
        pso = pso.append(temp)

    pso.index = range(len(pso))

    pso.to_sql(name="pso_data", con=engine, index=False, if_exists="replace", chunksize=10)

# Helpful function to see the details of a dataframe
def print_details(data):
    print("Number of customers: {}".format(data.customer_id.nunique()))
    print("Number of orders: {}".format(data.order_id.nunique()))
    print("Number of products: {}".format(data.product_id.nunique()))
    print("First date: {}".format(data.order_timestamp.min()))
    print("Last date: {}".format(data.order_timestamp.max()))
    print("Columns in data:")
    print(list(data.columns))
    print(data.shape)
    print(data.head(10))
    print(data.tail(10))

