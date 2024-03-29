import sys
num1 = float(sys.argv[1])
num2 = float(sys.argv[2])
 
# printing the sum in integer
print("Speed up from parallelism in the framework: {}x".format(round(num1 / num2, 2)))