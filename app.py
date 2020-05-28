from flask import Flask,request, url_for, redirect, render_template, jsonify, session
from pycaret.regression import *
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import sqlalchemy
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import SelectField, StringField
from forms import CarSelection


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cars.db'
app.config['SECRET_KEY'] = 'secret'
db = SQLAlchemy(app)

class Cars(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.String(50))
    make = db.Column(db.String(50))
    model = db.Column(db.String(50))

class Form(FlaskForm):
    year = SelectField('year', choices=[('1992', '1992'), ('1993', '1993'), ('1994', '1994'), ('1995', '1995'), ('1996', '1996'), ('1997', '1997'), ('1998', '1998'), ('1999', '1999'), ('2000', '2000'), ('2001', '2001'), ('2002', '2002'), ('2003', '2003'), ('2004', '2004'), ('2005', '2005'), ('2006', '2006'), ('2007', '2007'), ('2008', '2008'), ('2009', '2009'), ('2010', '2010'), ('2011', '2011'), ('2012', '2012'), ('2013', '2013'), ('2014', '2014'), ('2015', '2015'), ('2016', '2016'), ('2017', '2017'), ('2018', '2018'), ('2019', '2019'), ('2020', '2020'), ('2021', '2021')])
    brand = SelectField('brand', choices=[])
    kind = SelectField('kind', choices=[])
    mileage = StringField('Mileage:', [], render_kw={"placeholder": "Enter car mileage"})

@app.route('/', methods=['GET', 'POST'])
def home():
    form = CarSelection()

    if session.get('data'):
        data=session['data']
        return render_template('index.html',form=form,pred=data['pred'],cards=data['cards'],cards1=data['cards1'],cards2=data['cards2'])

    else:
        return render_template('index.html', form=form)

@app.route('/brand/<year>')
def brand(year):
    brands = Cars.query.filter_by(year=year).all()

    brandArray = []

    for brand in brands:
        brandObj = {}
        brandObj['model'] = brand.model
        brandObj['make'] = brand.make
        brandArray.append(brandObj)

    L = brandArray
    unique = list({v['make']:v for v in L}.values())
    a_unique = sorted(unique, key=lambda k: k['make'])

    return jsonify({'brands' : a_unique})

@app.route('/kind/<brand>/<year>')
def kind(brand, year):
    kinds = Cars.query.filter_by(make=brand, year=year).all()

    kindArray = []

    for kind in kinds:
        kindObj = {}
        kindObj['model'] = kind.model
        kindArray.append(kindObj)

    F = kindArray
    special = list({b['model']:b for b in F}.values())
    a_special = sorted(special, key=lambda k: k['model'])
    print(a_special)
    return jsonify({'kinds' : a_special})

model = load_model('catboost_8283')
cols = ['Year', 'Mileage', 'Make', 'Model']


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen, round = 2)
    prediction = int(prediction.Label[0])
    prediction_low = prediction - (prediction * 0.1)
    prediction_high = prediction + (prediction * 0.1)
    empty = []
    year = int_features[0]
    make = int_features[2]
    car_model = int_features[3]


############################################### Car Max ###############################################

    cards = []
    c_url = f'https://www.carmax.com/cars?search={make}+{car_model}'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
    c_response = requests.get(c_url, headers=headers, allow_redirects=False)
    c_html = c_response.text
    c_soup = BeautifulSoup(c_html, 'html.parser')
    c_results = c_soup.find_all('div', class_='car-tile')

    if len(c_results) > 5:
        c_results = c_results[0:5]

    if len(c_results) != 0:

        for c_result in c_results:

            c_link_page = c_result.find('div', {'class': 'orig'})['href']

            c_model_ = c_result.find('span', class_="model-trim").text
            # Identify and return price of listing
            c_price_ = c_result.find('span', class_="price").text
            # Identify and return mileage to listing
            c_miles_ = c_result.find('span', class_="miles").text
            # Identify and return link to listing
            c_link_ = c_result.find('div', class_='carousel-item hero-image').img['src']

            #mileage to go into
            c_miles_int = c_miles_.replace('K', '000')
            c_miles_int = c_miles_int.split(" ")
            c_miles_int = int(c_miles_int[0])

            c_predict_features = [year, c_miles_int, make, c_model_]

            c_data_unseen = pd.DataFrame([c_predict_features], columns=cols)

            c_prediction = predict_model(model, data=c_data_unseen, round = 2)

            c_prediction = c_prediction.Label

            card = f'''<div class="card" style="width: 45rem;" id='five_cards'>
                  <img class="card-img-top" src="{c_link_}" alt="Card image cap">
                  <div class="card-body">
                  <h5 class="card-title">{c_model_}</h5>
                  <p class="card-text">{c_miles_} {c_price_}</p>
                  <p class="card-text">Predicted Amount: ${c_prediction[0]:,.0f}</p>
                  <a href={c_link_page} class="btn btn-primary" target="_blank">Go to CarMax</a>
                  </div>
                  </div>'''

            cards.append(card)

            cardString = ' '.join(cards)
    else:
        cardString = '<br>'

############################################### Auto Trader ###############################################

    #Grab url and soup it
    at_url = f"https://www.autotrader.com/cars-for-sale/{year}/{make}/{car_model}/?makeCodeList={make}&searchRadius=0&marketExtension=include&startYear={year}&endYear={year}&modelCodeList={car_model}"
    at_response = requests.get(at_url, headers=headers, allow_redirects=False)
    at_html = at_response.content
    at_soup = BeautifulSoup(at_html, 'html')
    at_results = at_soup.find_all('div', {'data-cmp':'inventoryListing'})

    #Grab only first five listings
    if len(at_results) > 5:
        at_results = at_results[0:5]

    #Where cards get appended too
    cards1 = []

    #If there are results do for loop
    if (len(at_results)) != 0:

        #Loop through list of listings
        for i in at_results:

            #Gets each listing data
            div1 = i.find('div', {'class': 'padding-0 panel-body'})
            div2 = div1.find('div', {'data-cmp': 'itemCard'})
            title = div1.find('h2').text
            price = div1.find('span', {'class': 'first-price'}).text
            mileage = div1.find_all('div', {'class': "text-bold"})
            mileage = mileage[1]

            #Gets listing link
            a = div1.find('a', {'rel': 'nofollow'})['href']
            a = a.split('&')
            a = a[0]
            listing_a = f"https://www.autotrader.com{a}"
            response = requests.get(listing_a, headers=headers, allow_redirects=False)
            html = response.content
            soup = BeautifulSoup(html, 'html')

            #Grabs picture Link
            try:
                src = soup.find('img', {'class': 'img-responsive-scale'})['src']
            except:
                continue

            #Creates mileage to predict
            try:
                mileage_pred = mileage.split(' ')
                mileage_pred = int(mileage_pred[0].replace(',', ''))
                print(mileage_pred)
            except:
                mileage_pred = 10000

            #Creates car model feature to predict
            try:
                model_pred = title.split(make,1)[1]
            except:
                model_pred = car_model

            #creates predictions for auto trader listings
            at_predict_features = [year, mileage_pred, make, model_pred]
            at_data_unseen = pd.DataFrame([at_predict_features], columns=cols)
            at_prediction = predict_model(model, data=at_data_unseen, round = 2)
            at_prediction = at_prediction.Label[0]

            # Card for a listing
            card1 = f'''<div class="card" style="width: 45rem;" id='five_cards'>
                  <img class="card-img-top" src={src} alt="Card image cap">
                  <div class="card-body">
                  <h5 class="card-title">{title}</h5>
                  <p class="card-text">{mileage} ${price}</p>
                  <p class="card-text">Predicted Amount: ${at_prediction:,.0f}</p>
                  <a href="{listing_a}" class="btn btn-primary" target="_blank">Go to AutoTrader</a>
                  </div>
                  </div>'''

            #Creates list of cards
            cards1.append(card1)

            #Makes one long string of cards that gets put into card deck (Bootstrap)
            card1String = ' '.join(cards1)

    #If no results put down a br tag
    else:
        card1String = '<br>'

############################################### Cars Direct ###############################################

    # Gets Car Direct url and makes list of listings
    CD_url = f'https://www.carsdirect.com/used_cars/listings/{make}/{car_model}?zipcode=&dealerId=&distance=&yearFrom={year}&yearTo={year}&priceFrom=&priceTo=&qString={make}%603%6023%600%600%60false%7C{car_model}%604%60124%600%600%60false%7C&keywords=&makeName=make&modelName={car_model}&sortColumn=&sortDirection=&searchGroupId=&lnk='
    CD_response = requests.get(CD_url, headers=headers)
    CD_html = CD_response.text
    CD_soup = BeautifulSoup(CD_html, 'html.parser')
    CD_results = CD_soup.find_all('div', class_='list-row')

    print(CD_url)

    # Creates list to append cards
    cards2 = []

    if len(CD_results) > 5:
        CD_results = CD_results[0:5]

    print(CD_results)

    if (len(CD_results)) != 0:

        for CD_result in CD_results:
        # Error handling
            # Identify and return title of listing
            CD_link_page = CD_result['data-listinglink']
            CD_link_page = f"https://www.carsdirect.com{CD_link_page}"
            # Identify and return link of listing
            CD_make_ = CD_result.find('input', {'name': 'socialMake'})['value']
            # Identify and return make of listing
            CD_model_ = CD_result.find('input', {'name': 'socialModel'})['value']
            # Identify and return model of listing
            CD_year_ = CD_result.find('input', {'name': 'socialYear'})['value']
            # Identify and return price of listing
            CD_price_ = CD_result.find('input', {'name': 'priceLowLastUpdated'})['value']
            # Identify and return mileage to listing
            CD_miles_ = CD_result.find('div', {'class': 'mileage'}).text
            # Identify and return link to listing
            CD_link_ = CD_result.find('a', class_='list-img').img['src']

            try:
                mileage_pred = CD_miles_.split(' ')
                mileage_pred = int(mileage_pred[0].replace(',', ''))
                print(mileage_pred)
            except:
                mileage_pred = 10000

            CD_predict_features = [CD_year_, mileage_pred, CD_make_, CD_model_]
            CD_data_unseen = pd.DataFrame([CD_predict_features], columns=cols)
            CD_prediction = predict_model(model, data=CD_data_unseen, round = 2)
            CD_prediction = CD_prediction.Label[0]

            card2 = f'''<div class="card" style="width: 45rem;" id='five_cards'>
                  <img class="card-img-top" src={CD_link_} alt="Card image cap">
                  <div class="card-body">
                  <h5 class="card-title">{CD_year_} {CD_make_} {CD_model_}</h5>
                  <p class="card-text">{CD_miles_} Price: ${CD_price_}</p>
                  <p class="card-text">Predicted Price: ${CD_prediction:,.0f} </p>
                  <a href={CD_link_page} class="btn btn-primary" target="_blank">Go To Cars Direct</a>
                  </div>
                  </div>'''

            cards2.append(card2)

            card2String = ' '.join(cards2)
    else:
        card2String = '<br>'

    print(card2String)

    session['data'] = {'pred':'Expected Price will be between ${:,.2f}'.format(prediction_low) + ' and ${:,.2f}'.format(prediction_high),
    'cards':cardString,
    'cards1':card1String,
    'cards2':card2String
    }

    return redirect('/')

@app.route('/tableau')
def tableau():

    return render_template('tableau.html')

@app.route('/machine-learning')
def machine_learning():

    return render_template('analysis.html')

@app.route('/web-scraping')
def web_scraping():

    return render_template('web.html')

@app.route('/testimonial')
def testimonial():

    return render_template('test.html')


if __name__ == '__main__':
    app.run(debug=False)
