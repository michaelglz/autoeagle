from flask_wtf import FlaskForm
from wtforms import StringField, SelectField

class CarSelection(FlaskForm):
    """Car Selection."""

    year = SelectField('Year:', choices=[('', '--Select a year--'),('1992', '1992'), ('1993', '1993'), ('1994', '1994'), ('1995', '1995'), ('1996', '1996'), ('1997', '1997'), ('1998', '1998'), ('1999', '1999'), ('2000', '2000'), ('2001', '2001'), ('2002', '2002'), ('2003', '2003'), ('2004', '2004'), ('2005', '2005'), ('2006', '2006'), ('2007', '2007'), ('2008', '2008'), ('2009', '2009'), ('2010', '2010'), ('2011', '2011'), ('2012', '2012'), ('2013', '2013'), ('2014', '2014'), ('2015', '2015'), ('2016', '2016'), ('2017', '2017'), ('2018', '2018'), ('2019', '2019'), ('2020', '2020'), ('2021', '2021')])
    mileage = StringField('Mileage:', [], render_kw={"placeholder": "Enter car mileage"})
    brand = SelectField('Make:', choices=[('', '--Select a make--')])
    kind = SelectField('Model:', choices=[('', '--Select a model--')])
