from wtforms import Form, BooleanField, StringField, IntegerField, FloatField, validators


class GAForm(Form):
    equation = StringField(validators=[validators.DataRequired(), validators.Length(min=1)])
    num_generations = IntegerField(validators=[validators.DataRequired(), validators.NumberRange(min=0)])
    sol_per_pop = IntegerField(validators=[validators.DataRequired(), validators.NumberRange(min=1)])
    num_genes = IntegerField(validators=[validators.DataRequired(), validators.NumberRange(min=0)])
    accuracy = FloatField(validators=[validators.DataRequired()])
    mutation_probability = FloatField(validators=[validators.DataRequired(), validators.NumberRange(min=0, max=1)])
    num_parents_mating = IntegerField(validators=[validators.DataRequired()])
    crossover_type = StringField(validators=[validators.DataRequired()])
    parallel_processing = BooleanField()
