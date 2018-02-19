import time
class animal(object):
    def run(self):
        print"animal is runing"
class dog(animal):
    def __init__(self,h):
        self.h=h
    def run(self):
        print "dog is runing"
def set_age(self,age):
    self.age=age
from types import MethodType

a=animal()
a.set_age=MethodType(set_age,a)
a.set_age('eme')
print a.age
def set_score(self,score):
    self.score=score
animal.set_score=set_score
a.set_score(99)
print a.score
class student(object):
    def __init__(self,birth):
        self._birth=birth
    @property
    def birth(self):
        return self._birth

    @birth.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2015 - self._birth
b=student(100)
#b.birth(1888)
#print b.birth()
try:
    print "try"
    a=10/3
    print a
except ZeroDivisionError as e:
    print "ZeroDivisionError"
except ValueError as e:
    print "ValueError"
finally:
    print"finally..."
print"END"
ti=time.clock()
print ti
print"###########################"
class FooError(ValueError):
    pass
def foo(s):
    n=int(s)
    if n==0:
        raise FooError('invalid value:%s'%s)
    return 10/n
foo('1')
print"###########################"
def foo(s):
    n=int(s)
    assert n!=0,'hhhh'
    return 10/n
def main():
    foo('1')
    a=input('emmm')
main()
s=0
for i in range(50):
    s=s+1
ti=time.clock()-ti
print ti