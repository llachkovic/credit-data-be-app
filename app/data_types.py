import pandas as pd


class DataPoint:
    def __init__(self, df: pd.DataFrame):
        data: pd.Series = df.iloc[0]
        self.creditAmount = data.get('creditAmount', 0)
        self.installmentRate = data.get('installmentRate', 0)
        self.age = data.get('age', 0)
        self.checkingAccountStatus_A13 = data.get('checkingAccountStatus_GreaterThanOrEqual200DM', False)
        self.checkingAccountStatus_A14 = data.get('checkingAccountStatus_NoCheckingAccount', False)
        self.creditHistory_A34 = data.get('creditHistory_DelayInPaying', False)
        self.purpose_A41 = data.get('purpose_CarNew', False)
        self.purpose_A42 = data.get('purpose_CarUsed', False)
        self.purpose_A43 = data.get('purpose_FurnitureEquipment', False)
        self.savingsAccount_A61 = data.get('savingsAccount_LessThan100DM', False)
        self.savingsAccount_A62 = data.get('savingsAccount_Between100And500DM', False)
        self.savingsAccount_A63 = data.get('savingsAccount_Between500And1000DM', False)
        self.savingsAccount_A64 = data.get('savingsAccount_GreaterThanOrEqual1000DM', False)
        self.savingsAccount_A65 = data.get('savingsAccount_Unknown', False)
        self.employmentDuration_A71 = data.get('employmentDuration_Unemployed', False)
        self.employmentDuration_A72 = data.get('employmentDuration_LessThan1Year', False)
        self.employmentDuration_A73 = data.get('employmentDuration_Between1And4Years', False)
        self.employmentDuration_A74 = data.get('employmentDuration_Between4And7Years', False)
        self.employmentDuration_A75 = data.get('employmentDuration_GreaterThanOrEqual7Years', False)
        self.personalStatus_A93 = data.get('personalStatus_MaleSingle', False)
        self.property_A121 = data.get('property_RealEstate', False)
        self.property_A122 = data.get('property_BuildingSociety', False)
        self.property_A123 = data.get('property_CarOrOther', False)
        self.property_A124 = data.get('property_Unknown', False)
        self.otherInstallmentPlans_A143 = data.get('otherInstallmentPlans_None', False)
        self.housing_A151 = data.get('housing_Rent', False)
        self.housing_A152 = data.get('housing_Own', False)
        self.housing_A153 = data.get('housing_ForFree', False)
        self.telephone_A191 = data.get('telephone_None', False)
        self.telephone_A192 = data.get('telephone_Registered', False)


def one_hot_encode_payload(df, column):
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df.drop(column, axis=1, inplace=True)
    return df
