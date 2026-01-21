from locust import HttpUser, task, between

class SentimentUser(HttpUser):
    wait_time = between(1, 3)  # Simulate user thinking time 1-3 seconds

    @task(1)
    def health_check(self):
        self.client.get("/")

    @task(5)
    def predict_sentiment(self):
        # A set of test sentences
        formatted_data = {"text": "The company reported strong earnings growth."}
        self.client.post("/predict", json=formatted_data)

    @task(3)
    def predict_negative(self):
        formatted_data = {"text": "The stock market crashed significantly today."}
        self.client.post("/predict", json=formatted_data)
