MethaneScope - Streamlit deployment

1) Create a GitHub repository and upload:
   - MethaneScope.py
   - requirements.txt

2) Open https://share.streamlit.io/ and sign in with GitHub.
3) Click Create app.
4) Select the GitHub repository, branch (usually main), and file:
   MethaneScope.py
5) Optionally choose a custom app URL, then Deploy.
6) Share the resulting https://....streamlit.app link.

Copernicus authentication:
- Searching the public STAC catalog does not require a Copernicus login.
- To download/process Sentinel-2 data, each user clicks "Login & connect" in the app.
- The app sends the entered credentials directly to the official Copernicus identity endpoint over HTTPS.
- The app does not save the password; it keeps only the temporary access/refresh token in that user's Streamlit session.
- The "Open Copernicus website" button is also available for users who want to open the official CDSE site.

Important:
A login performed in the separate Copernicus website cannot automatically transfer its browser session/cookie to the Streamlit app. That is a browser security boundary. Therefore the app uses the official CDSE token endpoint for its own API session.
