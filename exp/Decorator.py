import builtins
"""
将print改写为当前函数可以使print的结果输出到指定的文件中，追加模式
注意：要和del print配合将print改为原来的print，否则所有print会输出到文件中
示例：
print = print_to_file(print, "this is first line", "./test.txt")
print("Hello, decorated world!")  
del print或print = builtins.print
print("Hello, original world!")
"""
def print_to_file(func, file_path, start_senctence=None):
    if start_senctence != None:
        with open(file_path, "a") as f:
            f.write(start_senctence + "\n")
    def wrapper(*args, **kwargs):
        with open(file_path, "a") as f:
            sep = kwargs.get("sep", " ")
            output_str = sep.join(str(arg) for arg in args)
            f.write("  "+output_str+"\n")
        func(*args, **kwargs)
        return 
    return wrapper

class mydsa():
    def __init__(self):
        pass
    def owww(self, func):
        func("owww print")
    
    def forward(self):
        print = print_to_file(builtins.print, "./test.txt")
        print("first print")
        self.owww(print)
        print = builtins.print
        print("last print")

if __name__ == "__main__":
    a = mydsa()
    a.forward()