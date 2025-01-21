docker:
	docker build . -t us.gcr.io/policyengine-api-prototype/gcf/us-central1/salt-amt --platform linux/amd64
	docker image push us.gcr.io/policyengine-api-prototype/gcf/us-central1/salt-amt
