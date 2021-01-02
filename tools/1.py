def foo(idx):
    print("starting...")
    for i in range(5):
        for j in range(3):
            print(idx, i, j)
            yield i


for i in foo(1):
    print(i)

for i in foo(2):
    print(i)