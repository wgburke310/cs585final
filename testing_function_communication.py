# Attempt to communicate with dataproc cluster and jupyter notebook scripts via local machine using an API key
import os

# Google imports
from google.oauth2 import service_account
from googleapiclient import discovery
from google.cloud import storage as dns


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cs585-final-project-b611a901a4b2.json"

# Load json info into variable
gcp_sa_credentials = {
    "type": "service_account",
    "project_id": "cs585-final-project",
    "private_key_id": "b611a901a4b2d8526a4ae8e547f0916d858ef219",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDgXM/p+h104eg1\nncwkNzx1xEfdrU5qsfRSHxqvMSEEI1yRdBGH+eotQT3Z1BTArT72GlhxywhbGIGv\nlihnSEYvW83VdIjJNEu25D2HFoTzyVAs5V4KZa/BtO08fU+SUmlae4TSiVqg3Kys\niHtJiq7OIeoSiiMQ8lGEZu0TLqAskccaG2H6UPcg27X6VVGwPiHmlrehdbFPZiAw\ntuDsOwVskH9syADZQAbvqgvokTpOLsLHxyIXtWSrEiMhqTbQLffecz7jNzVtBWsn\nVLP+j6+gWwHGFwRPZ2xxgPndwjjOsBT4ioosuMG/O1kaT231cxgfh2XPKdgkQgrz\nF2ZyWSr5AgMBAAECggEAF5C4B8/M8z9dF/CUhgNFNutgTwDKeTtquYtpzpfe62PZ\nb6+cvcb6mTk+iVgUu+4WbFv1CTK1lHfc+zfO05ZMROIkGvTr/xIel72eVsd/PGnb\nIXQP7hCjJrndpxpUVr6QMUBDAagKnFXBTzfglydZV+5I+xyGNVv2Qu1ankap7TRK\nNsDa3qOv5UfrnvQ521fkOLZDnvCmWhX/zxCe+eXtPCGeawLy4DGoGlszHyG3IIRW\nb3a44lQ9cZfHNBSF2L6AjDBoCRee005K5fMgzBOzilzK24VDH/ByQXJwTYOStP8z\n9Rm4BHngqA3CHAJdu6/rlxCFXAO/gtJ5ofWkELBUmQKBgQDzrIDhEE5T1SeUfABH\nl4KRb0r7pXsyhQnd2BiGJsozEhbsKSMJYPPOAvWIgGpfaXJJz0UJFFeYlgjoGAD2\nzCFXPLOQ/r3aSKYv24vIvGaluEpKa9OvmXJaVrCw3eDR9VPUvt8PyjoeeDgMZb1L\nIot1zLrv5uPshciNJjEykpqLvQKBgQDrtjvZdvDvF0WkULlPzrIr3LbsV8EZgyV2\nil0CcYFw207N8iDHFHV+HepUICN/kFpBJ8V0hKd2ESj2Kpl4OBpXAaCUasgESWQA\nxJjJEqZPZPTyBwIE6UvF2Y5M1Pem6Cpclxqngg37sziy7ohwXSLBqCpB7rP7/7DV\nFrGAwLtR7QKBgEGOx53FpU0oq91m/yxDtv0p1CKgAuU0pdLpsK9nAD99Pm2LI8IC\nM5XZdfWjlvrUg8sX2Jr4vhhvRTYsPxYnFVFDI4x+/NUddpipmJIJHhS34ETlfa5g\nukNTz3XGPBZAyq0SLTEyVzCbZ3juZl8wjBTFYAXrOKcJ10siW1of1zSBAoGBANo3\nxubID7w5vlamCTmScS7cUlkx0OqOmKNK61q0vJI/5pARZjkIftG4XlVtv498k6IY\ncNwzZ1mN/28O9y/uOKxuLDzbMruNOlDfsVcNtfxwybg7tqnXVjgf9na4/2F9NKKt\nnZaQd+ObA5Xb8WKdRu0kg6Kwm95j1FEihkhdpc3tAoGAUXbGTAqGtB7O8LZynCLo\n3OGWuu+qFKGdOcYn1kwYstrnggQp+clWm0dyeT/ma5dNJaJ8hrtO6Dk3WTGDVkXR\n3j14iUpzx2czbjYiqC3qiTSvBr6i3KdThwJ8BolYPEquot8fjmpXrVVFZq77DJ6v\n3YfveYIPaujofRCN/HoUccM=\n-----END PRIVATE KEY-----\n",
    "client_email": "local-machine@cs585-final-project.iam.gserviceaccount.com",
    "client_id": "106382637920719450067",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/local-machine%40cs585-final-project.iam.gserviceaccount.com",
}

project_id=gcp_sa_credentials["project_id"]

credentials = service_account.Credentials.from_service_account_info(gcp_sa_credentials)
client = dns.Client(project=project_id,credentials=credentials)

def my_function(request):
    return 'Hello World'

def hello(request): 
    if request.args:
        return f'Hello' + request.args.get('first') + request.args.get('last')+'\n'
    elif request.get_json():
        request_json = request.get_json()
        return  f'Hello' + request_json['first'] + request_json['last']+'\n'
    elif request.values:
        request_data = request.values
        return  f'Hello' + request_data['first'] + request_data['last']+'\n'
    else:
        return f'Hello World!'+'\n'

# Query with: curl http://localhost:8080?first=My&last=Name
# or on linux/mac: curl http://localhost:8080 -d ‘{“first”:”My”, “last”:”Name”}’
# or in browser: http://localhost:8080?first=My&last=Name