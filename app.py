from flask import Flask, render_template

app = Flask(__name__)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # ポート5000番でサーバーを起動
    app.run(host='0.0.0.0', port=5000, debug=True)