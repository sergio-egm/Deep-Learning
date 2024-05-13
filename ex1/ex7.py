class Polynomial:
    def __init__(self,deg):
        self.deg=deg
        self._par=[0.0] * (deg+1)
    
    def set_parameters(self,param):
        self._par=param
    
    def get_parameters(self):
        return self._par
    
    def _execute(self,x):
        r=self._par[0]
        d=1
        for a in self._par[1:]:
            r+=a*x**d
            d+=1
        return r
    
    def __call__(self,x):
        return self._execute(x)
    
    #@property
    # Valori ch possono essere visualizzati ma non modificati
    #@param.setter
    # posso definire un modo per modificare i valori

if __name__=="__main__":
    p=Polynomial(3)
    p.set_parameters([1,2,3])
    print(p.get_parameters())
    print(p(2))