import dataLoader
import FeatureBuilder
if __name__=='__main__':
   data = dataLoader.load_data()
   train_x,train_y,test_x,test_y = FeatureBuilder.build_feature(data)
