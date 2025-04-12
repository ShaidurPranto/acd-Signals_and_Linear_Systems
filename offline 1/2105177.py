import numpy as np
import matplotlib.pyplot as plt
import os

class Discrete_signal:

    def __init__(self,INF):
        self.INF = INF
        self.values = np.zeros(2*self.INF+1)

    def get_value_at_time(self,time):
        return self.values[time + self.INF]    

    def set_value_at_time(self,time,value):
        self.values[time + self.INF] = value
    
    def shift_signal(self,shift):
        new_signal = Discrete_signal(self.INF)
        new_signal.values = np.roll(self.values, shift)
        if shift > 0:
            new_signal.values[:shift] = 0
        elif shift < 0:
            new_signal.values[shift:] = 0
        return new_signal
    
    def add(self,other):
        new_signal = Discrete_signal(self.INF)
        new_signal.values = self.values + other.values
        return new_signal
    
    def multiply(self,other):
        new_signal = Discrete_signal(self.INF)
        new_signal.values = self.values * other.values
        return new_signal

    def multiply_const_factor(self,scaler):
        new_signal = Discrete_signal(self.INF)
        new_signal.values = self.values * scaler
        return new_signal

    def plot(self,label="",xlabel="",ylabel="",save_path=None):
        plt.stem(np.arange(-self.INF,self.INF+1),self.values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(label)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close() 


def get_discrete_unit_impulse_signal(INF):
    signal = Discrete_signal(INF)
    signal.set_value_at_time(0,1)
    return signal


class LTI_discrete:

    def __init__(self,impulse_response):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self,input_signal):
        output = []
        unit_impulse = get_discrete_unit_impulse_signal(input_signal.INF)
        for i in range(-input_signal.INF,input_signal.INF+1):
            if input_signal.get_value_at_time(i) != 0:
                shifted_unit_impulse = unit_impulse.shift_signal(i)
                output.append(shifted_unit_impulse.multiply_const_factor(input_signal.get_value_at_time(i)))
        
        return output
    
    def output(self,input_signal):
        output = []
        for i in range(-input_signal.INF,input_signal.INF+1):
            if input_signal.get_value_at_time(i) != 0:
                response = self.impulse_response.shift_signal(i)
                output.append(response.multiply_const_factor(input_signal.get_value_at_time(i)))
        
        return output



class Continuous_signal:

    def __init__(self,func):
        self.func = func
    
    def get_value_at_time(self,time):
        return self.func(time)

    def shift(self,shift):
        new_func = lambda x: self.func(x-shift)
        new_signal = Continuous_signal(new_func)
        return new_signal

    def add(self,other):
        new_func = lambda x: self.func(x) + other.func(x)
        new_signal = Continuous_signal(new_func)
        return new_signal

    def multiply(self,other):
        new_func = lambda x: self.func(x) * other.func(x)
        new_signal = Continuous_signal(new_func)
        return new_signal

    def multiply_const_factor(self,scaler):
        new_func = lambda x: self.func(x) * scaler
        new_signal = Continuous_signal(new_func)
        return new_signal
    
    def plot(self,INF = 10,span = 1000,label="",xlabel="",ylabel="",save_path=None):
        x = np.linspace(-INF,INF,span)
        y = self.func(x)
        plt.plot(x,y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(label)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close() 

    def plot_with_origin(self,origin,INF = 10,span = 1000,label="",xlabel="",ylabel="",save_path=None):
        x = np.linspace(-INF,INF,span)
        y = self.func(x)
        plt.plot(x,y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(label)

        origin_x = np.linspace(-INF,INF,span*100)
        origin_y = origin.func(origin_x)

        plt.plot(origin_x,origin_y,label="Original Line",color="orange")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close() 


def get_continuous_unit_impulse_signal(delta):
    def func(x):
        return np.where((x >= 0) & (x <= delta), 1 / delta, 0)
    return Continuous_signal(func)


class LTI_continuous:

    def __init__(self,impulse_response):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self,input_signal,delta,INF):
        output = []
        i = -INF
        unit_impulse = get_continuous_unit_impulse_signal(delta)
        while i <= INF:
            if input_signal.get_value_at_time(i) != 0: 
                shifted_unit_impulse = unit_impulse.shift(i)
                output.append(shifted_unit_impulse.multiply_const_factor(input_signal.get_value_at_time(i)))

            i = i + delta
        
        return output

    def output_approx(self,input_signal,delta,INF):
        output = []
        i = -INF
        while i <= INF:
            if input_signal.get_value_at_time(i) != 0: 
                response = self.impulse_response.shift(i)
                output.append(response.multiply_const_factor(input_signal.get_value_at_time(i)))

            i = i + delta
        
        return output
    
    def only_output(self,input_signal,delta,INF):
        output = []
        i = -INF
        while i <= INF:
            if input_signal.get_value_at_time(i) != 0: 
                response = self.impulse_response.shift(i)
                output.append(response.multiply_const_factor(input_signal.get_value_at_time(i)))

            i = i + delta
        
        output_signal = Continuous_signal(lambda x: 0)
        for signal in output:
            output_signal = output_signal.add(signal)

        return output_signal
    

#
# example functions - discrete
#

def show_discrete_LTI(INF = 10,save_path="discrete_1"):
    unit_impulse_response = Discrete_signal(INF)
    for i in range(-INF,INF+1):
        unit_impulse_response.set_value_at_time(i,np.exp(-i))
    lti = LTI_discrete(unit_impulse_response)
    input_signal = Discrete_signal(INF)
    for i in range(-INF,INF+1):
        input_signal.set_value_at_time(i,np.where(i < 0, 0, np.exp(-i)))

    input_signal.plot("Input Signal","Time","Amplitude",save_path + "/input_signal.png")

    decompositions_of_input_signal = lti.linear_combination_of_impulses(input_signal)
    reconstructed_input_signal = Discrete_signal(INF)
    for signal in decompositions_of_input_signal:
        signal.plot("Decomposition of Input Signal","Time","Amplitude",save_path + "/decomposition_of_input_signal.png")
        reconstructed_input_signal = reconstructed_input_signal.add(signal)
    reconstructed_input_signal.plot("Reconstructed Input Signal","Time","Amplitude",save_path + "/reconstructed_input_signal.png")

    unit_impulse_response.plot("Unit Impulse Response","Time","Amplitude",save_path + "/unit_impulse_response.png")

    impulse_respones = lti.output(input_signal)
    reconstructed_output_signal = Discrete_signal(INF)
    for signal in impulse_respones:
        signal.plot("Impulse Responses","Time","Amplitude",save_path + "/impulse_responses.png")
        reconstructed_output_signal = reconstructed_output_signal.add(signal)
    reconstructed_output_signal.plot("Reconstructed impulse responses","Time","Amplitude",save_path + "/reconstructed_impulse_responses.png")


def show_fixed_discrete_LTI(save_path="discrete_2"):
    impulse_response = Discrete_signal(5)
    impulse_response.set_value_at_time(0, 1)
    impulse_response.set_value_at_time(1, 1)
    impulse_response.set_value_at_time(2, 1)
    lti = LTI_discrete(impulse_response)
    input_signal = Discrete_signal(5)
    input_signal.set_value_at_time(0, 0.5)
    input_signal.set_value_at_time(1, 2)

    input_signal.plot("Input Signal", "Time", "Amplitude", save_path + "/input_signal.png")

    decompositions_of_input_signal = lti.linear_combination_of_impulses(input_signal)
    reconstructed_input_signal = Discrete_signal(5)
    for idx, signal in enumerate(decompositions_of_input_signal):
        signal.plot("Decomposition of Input Signal", "Time", "Amplitude", save_path + f"/decomposition_of_input_signal_{idx}.png")
        reconstructed_input_signal = reconstructed_input_signal.add(signal)
    reconstructed_input_signal.plot("Reconstructed Input Signal", "Time", "Amplitude", save_path + "/reconstructed_input_signal.png")

    impulse_response.plot("Unit Impulse Response", "Time", "Amplitude", save_path + "/unit_impulse_response.png")

    impulse_responses = lti.output(input_signal)
    reconstructed_output_signal = Discrete_signal(5)
    for idx, signal in enumerate(impulse_responses):
        signal.plot("Impulse Responses", "Time", "Amplitude", save_path + f"/impulse_responses_{idx}.png")
        reconstructed_output_signal = reconstructed_output_signal.add(signal)
    reconstructed_output_signal.plot("Reconstructed Impulse Responses", "Time", "Amplitude", save_path + "/reconstructed_impulse_responses.png")

#
# example functions - continuous
#

def show_continuous_LTI(delta=1, INF=10, span=100, save_path="continuous_1"):
    unit_impulse_response = Continuous_signal(lambda x: np.exp(-x))
    lti = LTI_continuous(unit_impulse_response)
    input_signal = Continuous_signal(lambda x: np.where(x < 0, 0, np.exp(-x)))

    input_signal.plot(10, 10000, "Input Signal", "Time", "Amplitude", save_path + "/input_signal.png")

    decompositions_of_input_signal = lti.linear_combination_of_impulses(input_signal, delta, INF)
    reconstructed_input_signal = Continuous_signal(lambda x: 0)
    for idx, signal in enumerate(decompositions_of_input_signal):
        signal.plot(INF, span, "Decomposition of Input Signal", "Time", "Amplitude", save_path + f"/decomposition_of_input_signal_{idx}.png")
        reconstructed_input_signal = reconstructed_input_signal.add(signal)
    reconstructed_input_signal.plot(INF, span, "Reconstructed Input Signal", "Time", "Amplitude", save_path + "/reconstructed_input_signal.png")

    unit_impulse_response.plot(INF, span, "Unit Impulse Response", "Time", "Amplitude", save_path + "/unit_impulse_response.png")

    impulse_responses = lti.output_approx(input_signal, delta, INF)
    reconstructed_output_signal = Continuous_signal(lambda x: 0)
    for idx, signal in enumerate(impulse_responses):
        signal.plot(INF, span, "Impulse Responses", "Time", "Amplitude", save_path + f"/impulse_responses_{idx}.png")
        reconstructed_output_signal = reconstructed_output_signal.add(signal)
    reconstructed_output_signal.plot(INF, span, "Reconstructed Impulse Responses", "Time", "Amplitude", save_path + "/reconstructed_impulse_responses.png")


def show_fixed_continuous_LTI(T=5, delta=1, INF=25, span=100, save_path="continuous_2"):
    unit_impulse_response = Continuous_signal(lambda x: np.where((x > 0) & (x < 2 * T), x, 0))
    lti = LTI_continuous(unit_impulse_response)
    input_signal = Continuous_signal(lambda x: np.where((x > 0) & (x < T), 1, 0))

    input_signal.plot(10, 10000, "Input Signal", "Time", "Amplitude", save_path + "/input_signal.png")

    decompositions_of_input_signal = lti.linear_combination_of_impulses(input_signal, delta, INF)
    reconstructed_input_signal = Continuous_signal(lambda x: 0)
    for idx, signal in enumerate(decompositions_of_input_signal):
        signal.plot(INF, span, "Decomposition of Input Signal", "Time", "Amplitude", save_path + f"/decomposition_of_input_signal_{idx}.png")
        reconstructed_input_signal = reconstructed_input_signal.add(signal)
    reconstructed_input_signal.plot(INF, span, "Reconstructed Input Signal", "Time", "Amplitude", save_path + "/reconstructed_input_signal.png")

    unit_impulse_response.plot(INF, span, "Unit Impulse Response", "Time", "Amplitude", save_path + "/unit_impulse_response.png")

    impulse_responses = lti.output_approx(input_signal, delta, INF)
    reconstructed_output_signal = Continuous_signal(lambda x: 0)
    for idx, signal in enumerate(impulse_responses):
        signal.plot(INF, span, "Impulse Responses", "Time", "Amplitude", save_path + f"/impulse_responses_{idx}.png")
        reconstructed_output_signal = reconstructed_output_signal.add(signal)
    reconstructed_output_signal.plot(INF, span, "Reconstructed Impulse Responses", "Time", "Amplitude", save_path + "/reconstructed_impulse_responses.png")


def show_fixed_continuous_LTI_2():
    T=5
    delta=2
    INF=25
    span=100
    save_path="continuous_3"

    unit_impulse_response = Continuous_signal(lambda x: np.where((x > 0) & (x < 2 * T), x, 0))
    lti = LTI_continuous(unit_impulse_response)
    input_signal = Continuous_signal(lambda x: np.where((x > 0) & (x < T), 1, 0))

    input_signal.plot(10, 10000, "Input Signal", "Time", "Amplitude", save_path + "/input_signal.png")

    decompositions_of_input_signal = lti.linear_combination_of_impulses(input_signal, delta, INF)
    reconstructed_input_signal = Continuous_signal(lambda x: 0)
    for idx, signal in enumerate(decompositions_of_input_signal):
        signal.plot(INF, span, "Decomposition of Input Signal", "Time", "Amplitude", save_path + f"/decomposition_of_input_signal_{idx}.png")
        reconstructed_input_signal = reconstructed_input_signal.add(signal)
    reconstructed_input_signal.plot(INF, span, "Reconstructed Input Signal", "Time", "Amplitude", save_path + "/reconstructed_input_signal.png")

    unit_impulse_response.plot(INF, span, "Unit Impulse Response", "Time", "Amplitude", save_path + "/unit_impulse_response.png")

    impulse_responses = lti.output_approx(input_signal, delta, INF)
    reconstructed_output_signal = Continuous_signal(lambda x: 0)
    for idx, signal in enumerate(impulse_responses):
        signal.plot(INF, span, "Impulse Responses", "Time", "Amplitude", save_path + f"/impulse_responses_{idx}.png")
        reconstructed_output_signal = reconstructed_output_signal.add(signal)
    reconstructed_output_signal.plot(INF, span, "Reconstructed Impulse Responses", "Time", "Amplitude", save_path + "/reconstructed_impulse_responses.png")

    #comparing with actual output
    actual_output_signal = lti.only_output(input_signal, 1 * 0.1 , INF)
    new_delta = delta
    for i in range(4):
        new_delta = new_delta / 2
        output_signal = lti.only_output(input_signal, new_delta, INF)
        output_signal.plot_with_origin(actual_output_signal, INF, span, f"Output Signal with delta = {new_delta}", "Time", "Amplitude", save_path + f"/output_signal_{i}.png")



def main():
    a = 5
    # show_discrete_LTI()
    # show_fixed_discrete_LTI()
    # show_continuous_LTI()
    # show_fixed_continuous_LTI()
    show_fixed_continuous_LTI_2()

main()