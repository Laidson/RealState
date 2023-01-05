


class MetricCalulation:
    
    def __init__(self) -> None:
        pass

    def num_of_propeties(self, df):
        """number of lines on the data frame"""
        return len(df)

    def avg_price(self, df):
        """Recive a column named PRICE in format 'numpy.int64'"""
        return "${:,}".format(df['PRICE'].mean()).split(',')[0] + 'K'

    def avg_dom(self, df):
        """Recive a column named DAYS ON MARKET in numpy.float64"""
        return int(df['DAYS ON MARKET'].mean())
    
    def avg_ppsqft(self,df):
        """Recive a column named $/SQUARE FEET format numpy.float64"""
        return "${:,}".format(int(df['$/SQUARE FEET'].mean()))

    def add_bd(self, df):
        """Number of properties with additonal bedroom opportunity"""
        return df.loc[df['additional_bd_opp'] == True]
    
    def add_adu(self, df):
        'Number of properties with ADU potential'
        return df.loc[df['adu_potential'] == True]