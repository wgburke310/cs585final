# cs585final

Here is the code we used for our final project for cs 585. Our presentation and full report including results were submitted on canvas.

After many hours of trying to get Docker to work, we ran into many permission errors and persistence problems with containers shutting down immediately after startup. While the docker container does not remain running after startup, it can be created using "docker-compose up -d". An example file that would be run in the container is query_function.ipynb. This notebook can be run independently and will receive a response from the google cloud server. This function queries a Google Cloud Function which in turn accesses our bucket where we store our data and runs our machine learning model's evaluation script and returns the results to the executor. 

Other files in this repository include testing files for connecting to Google Cloud services, as well as the scripts to generate our models (binary_classification.py). 
