from metaflow import FlowSpec, step, Flow, Parameter, JSONType
import numpy as np
import json


class PredictFlow(FlowSpec):
    vector = Parameter('vector', type=JSONType, required=True, help='Input features as a JSON array')

    @step
    def start(self):
        run = Flow('TrainFlow').latest_run
        self.train_run_id = run.pathspec
        self.model = run['end'].task.data.model
        
        if isinstance(self.vector, str):
            self.vector = json.loads(self.vector)
        
        self.features = np.array(self.vector).reshape(1, -1)
        print(f"Input features: {self.features}")
            
        self.next(self.predict)

    @step
    def predict(self):
        self.prediction = self.model.predict(self.features)[0]
        print(f"Prediction successful: {self.prediction}")
            
        self.next(self.end)

    @step
    def end(self):
        print(f"Model: {type(self.model).__name__}")
        print(f"Predicted value: {self.prediction:.2f}")
        print(f"Input features: {self.features[0]}")


if __name__ == '__main__':
    PredictFlow()
