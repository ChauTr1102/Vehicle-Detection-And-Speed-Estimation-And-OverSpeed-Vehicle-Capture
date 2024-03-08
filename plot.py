import matplotlib.pyplot as plt

# Speed data from the program
program_speed = [28.8, 30, 25, 20, 27, 34, 21, 36, 32]

# Real speedometer data
speedometer_data = [30, 32, 23, 20, 28, 31, 21, 36, 33]

# Plotting
plt.plot(program_speed, label='Program Speed Estimation', marker='o')
plt.plot(speedometer_data, label='Real Speedometer Data', marker='x')
plt.xlabel('Sample')
plt.ylabel('Speed')
plt.title('Program vs Real Speedometer Data')
plt.legend()
plt.show()
