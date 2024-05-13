def ave(x):
    return sum(x)/len(x)

def fun(x):
    appo=1
    num=0
    while x>num:
        num+=1
        appo*=num
    return appo

if __name__=="__main__":
    v=[2,6,3,8,9,11,-5]
    print(f'Punto 1 {ave(v)}')
    print(f'Punto 2 {fun(5)}')