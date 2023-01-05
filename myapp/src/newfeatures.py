

class NewFeatures:
    def __init__(self) -> None:
      pass

    def ratio_square_feet_dedroom(self, df):
        try:
            result = df['SQUARE FEET'] / df['BEDS']
            return result
        except:
            return 'ERROR'

    def ratio_lot_square_feet(self, df):
        try:
            result = df['LOT SIZE'] / df['SQUARE FEET']
            return result
        except:
            return 'ERROR'
        

    def additional_bedroom_opportunity(self, df):
        """
            Additional bedroom — 
            Identify properties where an additional bedroom can be added to increase income. 
            This can include converting a den or dining room to a bedroom. 
            A property should have a large square footage relative to the number of bedrooms.

            df: data farme
            columns needed:
                ratio_sqft_bd
                ratio_sqft_bd
                BEDS
                PROPERTY TYPE
        """

        try:
            # 2bd >= 1300 can usually fit an additional bd
            # 3bd >= 1950 can usually fit an additional bd
            # 4bd >= 2600 can usually fit an additional bd
            if (df['ratio_sqft_bd'] >= 650) and (df['ratio_sqft_bd'] is not None) and (df['BEDS'] > 1) and (df['PROPERTY TYPE'] == 'Single Family Residential'):
                return True
            else:
                return False
        
        except:
            return False


    def adu_potential(self, df):
        """
            ADU (additional dwelling unit) — 
            Identify single-family homes that can accommodate a secondary housing unit. 
            This can include converting a basement or adding a mother-in-law suite.

             df: data farme
            columns needed:
                ratio_lot_sqft
                ratio_lot_sqft
                HOA/MONTH
                PROPERTY TYPE
        """

        try:
            if (df['ratio_lot_sqft'] >= 5) and (df['ratio_lot_sqft'] is not None) and (df['HOA/MONTH'] is not None) and (df['PROPERTY TYPE'] == 'Single Family Residential'):
                return True
            else:
                return False
        except:
            return False