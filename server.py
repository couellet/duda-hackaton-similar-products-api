import requests
import pandas as pd
import urllib.parse
import os
import socketserver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
 
    def do_GET(self):

        self.path

        qs = urllib.parse.parse_qs(self.path)

        product_id = qs.get('?productid')[0]
        url = 'https://product-recommendations-server.vercel.app/api/products'

        response = requests.get(url)
        data = response.json()
        
        newData = pd.json_normalize(data)

        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0.0, stop_words='english')
        tfidf_matrix = tf.fit_transform(newData['words'])

        # Calculate the dot product between the vectors
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        newData = newData.reset_index()
        titles = newData['id']
        indices = pd.Series(newData.index, index=newData['id'])

        # Get the index of the product that matches the title
        idx = indices[product_id]
    
        # Get the pairwise similarity scores of all products with that product
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the products based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the 30 most similar products
        sim_scores = sim_scores[1:31]
        
        print(sim_scores)
        # Get the product indices
        product_indices = [i[0] for i in sim_scores]
        
        # Return the titles corresponding to the filtered indices
        recommendations = titles.iloc[product_indices]

        json_output = recommendations.to_json(orient='records')

        print(json_output)
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
        self.wfile.write(str(json_output).encode('utf-8'))
        return

port = int(os.getenv('PORT', 80))
print('Listening on port %s' % (port))
httpd = socketserver.TCPServer(('', port), handler)
httpd.serve_forever()