a,b=(1,1)
target=48
for i in range(3,target+1):
    appo=a+b
    print(f'F({int(i)})= {appo}')
    a,b=(b,appo)
