web: sh setup.sh && streamlit run main.py
web: gunicorn -w 4 -b 0.0.0.0:$PORT -k gevent main:app
