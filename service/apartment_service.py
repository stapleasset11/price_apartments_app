import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from domain.domain import ApartmentRequest, ApartmentResponse

class ApartmentService():
    def __init__(self):
        self.path_model = "artifacts/randomForestForApartmentPrice.pkl"
        self.path_encoder = "artifacts/neighbourhood_encoder.pkl"
        self.model = self.load_artifact(self.path_model)
        self.le = self.load_artifact(self.path_encoder)

    def load_artifact(self,path_to_artifact):
        with open(path_to_artifact,'rb') as f:
            artifact = pickle.load(f)

        return artifact
    
    def preprocess_input (self,request:ApartmentRequest) -> pd.DataFrame:
        data_dict = {'rooms':request.rooms,
                     'size':request.size,
                     'bathrooms':request.bathrooms,
                     'neighbourhood':request.neighbourhood,
                     'year_built':request.year_built}

        data_df = pd.DataFrame.from_dict([data_dict])

        data_df.neighbourhood = data_df.neighbourhood.str.lower()
        data_df.neighbourhood  = self.le.transform(data_df.neighbourhood )
        data_df.neighbourhood  = data_df.neighbourhood.astype('category')
        return data_df
    
    def predict_price(self,request:ApartmentRequest) -> ApartmentResponse:
        input_df = self.preprocess_input(request)
        apartment_price = self.model.predict(input_df)[0]
        apartment_price = int(apartment_price)
        
        response = ApartmentResponse
        response.price = apartment_price
        return response
    
# if __name__ == "__main__":
#     test_request = ApartmentRequest(rooms = 120,size = 1200,bathrooms = 31,neighbourhood='manastur',year_built=2020)
#     apt_serve = ApartmentService()
#     res = apt_serve.predict_price(request=test_request)
#     print(res.price)

