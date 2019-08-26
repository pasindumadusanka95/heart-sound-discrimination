echo "Enter application name(eg. app.py): "
read name
export FLASK_APP=$name
flask run --host=0.0.0.0
