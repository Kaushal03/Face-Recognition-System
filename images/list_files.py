# import OS module
import os

# Get the list of all files and directories
path = r"C:\Users\Admin\AppData\Local\Programs\Python\Python310\My Programs\Face Recognition\images"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

# prints all files
print(dir_list)
