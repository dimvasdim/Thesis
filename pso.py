import neural_network as nn
from random import random, uniform
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine


# Credentials to connect to the database
username = "username"
password = "DB_password"
hostname = "DB_host"
dbname = "DB_name"


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.pbest_position = position
        self.gbest_position = position
        self.pbest_fitness = 0
        self.gbest_fitness = 0

    def evaluate(self, obj_function, steady, cost, model):
        fitness = obj_function(self.position, steady, cost, model)
        # Check the personal best
        if fitness > self.pbest_fitness:
            self.pbest_fitness = fitness
            self.pbest_position = self.position
        # Check the global best
        if fitness > self.gbest_fitness:
            self.gbest_fitness = fitness
            self.gbest_position = self.position

    def update_velocity(self, w, c1, c2):
        r1 = random()
        r2 = random()
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (self.gbest_position - self.position)
        self.velocity = (w * self.velocity) + cognitive + social

    def update_position(self, bounds):
        low_bound = bounds[0]
        high_bound = bounds[1]
        self.position = self.position + self.velocity
        # adjust minimum position if needed
        if self.position < low_bound:
            self.position = low_bound
        # adjust maximum position if needed
        if self.position > high_bound:
            self.position = high_bound


def particle_swarm(number_of_particles, c1, c2, w_min, w_max, iterations, obj_function, steady, cost, model, bounds):
    '''
        number_of_particles: the population of the particles that try to maximize a function
        c1: cognitive component of a particle
        c2: social component of a particle
        w_min: minimum inertia
        w_max: maximum inertia
        iterations: number of iterations to achieve convergence
        obj_function: the function that we try to maximize
        bounds: bounds for the position
        return: best position and maximum value of the function
    '''
    # Initialize the swarm
    swarm = []
    for i in range(number_of_particles):
        position = round(uniform(bounds[0], bounds[1]), 2)
        velocity = round(uniform(-0.5, 0.5), 2)
        swarm.append(Particle(position, velocity))


    for i in range(iterations):
        w = w_max - i * ((w_max - w_min) / iterations)
        for p in range(number_of_particles):
            swarm[p].evaluate(obj_function, steady, cost, model)
            swarm[p].update_velocity(w, c1, c2)
            swarm[p].update_position(bounds)

    best_position = swarm[number_of_particles - 1].gbest_position
    best_value = swarm[number_of_particles - 1].gbest_fitness

    return (best_position, best_value)


def gain(price, steady, cost, model):
    temp = np.array(steady)
    temp = np.append(temp, price)
    temp = np.reshape(temp, (1, len(temp)))
    quantity = np.round_(model.predict(temp))
    result = (price - cost) * quantity
    return float(np.round_(result, 2))


def optimize_prices():
    # Connect to the database of the e-shop
    engine = create_engine("mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
                           .format(user=username,
                                   password=password,
                                   host=hostname,
                                   dbname=dbname))

    data = pd.read_sql_table("pso_data", engine)
    opt_data = pd.read_sql_table("data_for_optimization", engine)

    product_encoder = nn.nn_final_training()
    data["product_id"] = product_encoder.transform(data["product_id"])
    model = load_model("final_model.h5")

    columns = ["product_id", "low_bound", "high_bound",
               "optimized_price", "predicted_quantity",
               "predicted_gain", "arxikiTimi", "telikiTimi"]
    optimization_results = pd.DataFrame(columns=columns)

    for index, row in data.iterrows():
        print(index)
        upper_bound = row.product_max_bound
        lower_bound = row.product_min_bound
        steady = row[2:-1]
        cost = row.product_cost
        bounds = (lower_bound, upper_bound)
        number_of_swarms = 40
        c1 = 0.4
        c2 = c1
        w_min = 0.6
        w_max = 0.8

        best_position, best_value = particle_swarm(number_of_swarms, c1, c2,
                                                   w_min, w_max, 400, gain,
                                                   steady, cost, model, bounds)
        temp = np.array(steady)
        temp = np.append(temp, best_position)
        temp = np.reshape(temp, (1, len(temp)))
        predicted_quantity = int(np.round_(model.predict(temp)))

        optimization_results.loc[index, "product_id"] = row["product_id"]
        optimization_results.loc[index, "low_bound"] = float(bounds[0])
        optimization_results.loc[index, "high_bound"] = float(bounds[1])
        optimization_results.loc[index, "optimized_price"] = float(round(best_position, 2))
        optimization_results.loc[index, "predicted_quantity"] = predicted_quantity
        optimization_results.loc[index, "predicted_gain"] = round(best_value, 2)

    optimization_results["product_id"] = product_encoder.inverse_transform(data["product_id"])
    optimization_results.to_sql(name="optimization_results", con=engine, index=False, if_exists="replace", chunksize=1)
    for index, row in optimization_results.iterrows():
        arxikiTimi = opt_data.loc[opt_data["product_id"] == row["product_id"]].arxikiTimi.values
        arxikiTimi = round(float(arxikiTimi), 2)
        telikiTimi = opt_data.loc[opt_data["product_id"] == row["product_id"]].telikiTimi.values
        telikiTimi = round(float(telikiTimi), 2)
        optimization_results.loc[index, "arxikiTimi"] = arxikiTimi
        optimization_results.loc[index, "telikiTimi"] = telikiTimi

    optimization_results.to_sql(name="optimization_results", con=engine, index=False, if_exists="replace", chunksize=1)


