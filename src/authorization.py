#Authorization for Spotify API: client id and client secret id not disclosed for privacy
import tekore as tk
def authorize():
 CLIENT_ID = "CLIENT-ID-HERE"
 CLIENT_SECRET = "CLIENT-SECRET-ID-HERE"
 app_token = tk.request_client_token(CLIENT_ID, CLIENT_SECRET)
 return tk.Spotify(app_token)
