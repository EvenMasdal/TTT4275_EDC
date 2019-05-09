classmapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

class petal:
    """docstring for petal"""
    def __init__(self, properties, classification):
        self.properties = properties
        self.classification = classification

    def __getitem__(key):
        return self.properties[key]

    def __repr__(self):
        return f'{self.properties}  |  {self.classification}'
        

class DataLoader:
    """docstring for DataLoader"""
    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        initial = []
        for line in lines:
            splt = line.split(',')
            properties = [float(i) for i in splt[:-1]]
            classification = splt[-1]
            initial.append(petal(properties, classification))

        self.data = []
        for i in range(50):
            for j in range(3):
                self.data.append(initial[50*j+i])

    def __str__(self):
        return str(self.data)

if __name__ == '__main__':
    a = DataLoader('iris.data')
    petal = a.data[0]

    print(petal)