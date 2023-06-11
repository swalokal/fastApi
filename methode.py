class toko(dict): 
    def __init__(self): 
        self.data = {
            "data": []
        }
           
    def add(self, result): 
        for item in result:
            data_item = {
                "id": item[0],
                "name": item[1],
                "latitude": item[2],
                "longitude": item[3]
            }
            self.data["data"].append(data_item)
class product(dict): 
    def __init__(self): 
        self.data = {
            "data": []
        }
           
    def add(self, result): 
        for item in result:
            data_item = {
                "id": item[0],
                "name": item[1],
                "price": item[2]
            }
            self.data["data"].append(data_item)
class allin(dict): 
    def __init__(self): 
        self.data = {
            "data": []
        }
           
    def add(self, result): 
        for item in result:
            data_item = {
                "id": item[0],
                "name": item[1],
                "price": item[2],
                "toko": item[3],
                "latitude": item[4],
                "longtitude": item[5]
            }
            self.data["data"].append(data_item)