import sys
num1 = float(sys.argv[1])
num2 = float(sys.argv[2])
 
# printing the sum in integer
if sys.argv[3] == 'par':
    print("Speed up from parallelism in the framework: {}x".format(round(num1 / num2, 2)))
if sys.argv[3] == 'lang':
    print("Speed up from the language difference     : {}x".format(round(num1 / num2, 2)))