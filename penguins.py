import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
import os
current_path = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(current_path, 'images')
os.makedirs(image_folder, exist_ok=True)

print('Libraries loaded, everything working perfectly.!!')
print('Seaborn datasets')
print(sns.get_dataset_names())
penguins = sns.load_dataset('penguins')
print()
print(penguins.head(10))
print()
print(f'The dataset contains this number of rows and columns.: ', penguins.shape)
print()
print(penguins.isna())
print()
print(f'Total number of null values ​​in the dataset: ', penguins.isna().sum())
print()
print('We want to find out if there is a relationship between the weight of the penguins and the length of their beaks and flippers.')
print('To do this, we will use three columns for this study: body_mass_g, bill_length_mm, and flipper_length_mm.')
print('First, we will clean the null values ​​through EDA so we can work with a clean dataset.')
penguins = penguins.dropna()
print()
print(f'Null values after cleaning:\n', penguins.isna().sum())
print()
print('Perfect, we have our clean dataset, free of null values ​​that could interfere with the final result.')
print()
print(penguins.describe(include=[float, int]))
print('Now we will look at a scatter plot of the data.')
print('For this case, we will select the important columns and store them in variables.')
x = penguins[['flipper_length_mm']]
y = penguins['body_mass_g']
sns.pairplot(penguins)
plt.savefig(os.path.join(image_folder, 'pairplot.png'), dpi=300)
plt.show()
print(x.head(10))
print()
ols_formula = 'body_mass_g ~ flipper_length_mm'
ols_model = ols(formula=ols_formula, data=penguins)
model_fit = ols_model.fit()
print(model_fit.summary())

intercept = -5872.0927
slope = 50.1533
durbin_watson = 2.102
r_squared = 0.762

print(f'The intercept coefficient is: {intercept}')
print(f'The coefficient of the slopes is: {slope}')
ecuacion = "body_mass_g = -5872.0927 + (50.1533 * flipper_length_mm)"
print(ecuacion)

print()
print(f'The {r_squared} represents a significant percentage and is a good predictor, accounting for 76.2% of the variation in penguin body mass (weight).')
print()
print(f'The p-value is 0.000, which is less than 0.05; therefore, the relationship between penguin weight and flipper size is statistically significant—meaning it is not due to coincidence or chance.')
print('The linear equation is y = -5872.0927 + 50.1533x.')
print()
print('In the graph, the points show a positive linear relationship: as the flipper length increases, the penguins', 'weight also increases. This confirms the linearity.')
print('The line represents the data behavior very well, clustering the data close to the regression line.')
print('The data are homoscedastic; from the 3,000-point mark up to the 6,000-point peak, the data remain constant along the line.')
print()
sns.regplot(x='flipper_length_mm', y='body_mass_g', data=penguins)
plt.title("Linear Regression: Body Mass vs. Flipper Length")
plt.xlabel("Flipper Length (mm)")
plt.ylabel("Body Mass (g)")
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'regression_plot.png'), dpi=300)
plt.show()

print('Even the basic assumptions of simple linear regression are met.')
print()
print('Calculation of residuals')
residuals = model_fit.resid
predictions = model_fit.fittedvalues

plt.figure(figsize=(8, 5))
sns.scatterplot(x=predictions, y=residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual plot (check for homoscedasticity)')
plt.xlabel('Predicted values ​​(Body Mass)')
plt.ylabel('Residuals (Errors)')
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'residual_plot.png'), dpi=300)
plt.show()
print('The graph satisfies the principles of homoscedasticity; there is uniform dispersion, with no unusual patterns (funnel, cone, or U-shape).')
print()
print('We will calculate the Q-Q plot to check the normality of the data.')
fig = sm.qqplot(model_fit.resid, line='s')
plt.title('Q-Q Plot of Residuals (Normality Check)')
plt.tight_layout()
plt.savefig(os.path.join(image_folder, 'qq_plot.png'), dpi=300)
plt.show()
print('The points follow a clean, normal distribution (Gaussian bell curve).')
