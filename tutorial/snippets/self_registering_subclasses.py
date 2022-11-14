class Vehicle:
    _kinds = {}

    def __init_subclass__(cls, /, kind: str, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._kinds[kind] = cls

    def __new__(cls, *args, kind: str, **kwargs):
        return super().__new__(cls._kinds[kind])
        
    def say_name(self):
        raise NotImplementedError

class Car(Vehicle, kind="car"):
    def __init__(self, value, **kwargs):
        self.value = value
        print("Initializing car")
        print(f"{kwargs=}")
    
    def say_name(self):
        print("I am a car")

class Truck(Vehicle, kind="truck"):
    def __init__(self, value, **kwargs):
        self.value = value
        print("Initializing truck")
        print(f"{kwargs=}")
    
    def say_name(self):
        print("Truck coming through")

class Semi(Truck, kind="semi"):
    def __init__(self, value, **kwargs):
        self.value = value
        print("Initializing semi")
        print(f"{kwargs=}")
    
    def say_name(self):
        print("Oh lawd, it's a SEMI")


if __name__ == "__main__":
    car = Vehicle(7, kind="car")
    car.say_name()

    truck = Vehicle(kind="truck", value = 9)
    truck.say_name()

    semi = Vehicle(kind="semi", value = 8)
    semi.say_name()
