"""Generates a tiny synthetic EN-NE corpus so the pipeline can be run
end to end without the real data.

These sentences are made up for demonstration - they are NOT from the NLTM
corpus, which isn't ours to redistribute. They're written to span all nine
domains so stage 1 routing and stage 2 fine-graining both have something to
chew on.

Writes two workbooks on purpose, with different column headers, because the
real source data is inconsistent that way (ID/src/tgt vs ID/SOURCE/TARGET vs
no header row at all) and merge_data.py reads by position to cope.

    python sample_data/make_sample.py
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    # Tech
    ("The application requires a stable internet connection.",
     "यो एप्लिकेसनलाई स्थिर इन्टरनेट जडान आवश्यक पर्छ।"),
    ("Install the latest software update before restarting the computer.",
     "कम्प्युटर पुनः सुरु गर्नुअघि नवीनतम सफ्टवेयर अपडेट इन्स्टल गर्नुहोस्।"),
    ("The database server stores user records securely.",
     "डाटाबेस सर्भरले प्रयोगकर्ताको विवरण सुरक्षित रूपमा राख्छ।"),
    ("Mobile network coverage has expanded to remote districts.",
     "मोबाइल नेटवर्क सेवा दुर्गम जिल्लासम्म विस्तार भएको छ।"),
    ("Developers released a patch to fix the security bug.",
     "विकासकर्ताहरूले सुरक्षा त्रुटि सच्याउन प्याच जारी गरे।"),
    # Agriculture
    ("Farmers in the hills cultivate millet and buckwheat.",
     "पहाडका किसानहरूले कोदो र फापर खेती गर्छन्।"),
    ("Use organic fertiliser to improve the soil quality.",
     "माटोको गुणस्तर सुधार गर्न प्रांगारिक मल प्रयोग गर्नुहोस्।"),
    ("The paddy harvest was delayed by the late monsoon.",
     "ढिलो मनसुनका कारण धान बाली भित्र्याउन ढिलाइ भयो।"),
    ("Livestock farming provides income to many rural families.",
     "पशुपालनले धेरै ग्रामीण परिवारलाई आम्दानी दिन्छ।"),
    ("Irrigation canals were repaired before the planting season.",
     "रोपाइँ सिजनअघि सिँचाइ कुलो मर्मत गरियो।"),
    # Climate
    ("The monsoon season brings heavy rainfall to the region.",
     "मनसुनले यस क्षेत्रमा भारी वर्षा ल्याउँछ।"),
    ("Glaciers in the Himalaya are retreating every year.",
     "हिमालयका हिमनदीहरू प्रत्येक वर्ष पछि हट्दै छन्।"),
    ("Carbon emissions must be reduced to limit global warming.",
     "विश्वव्यापी तापमान वृद्धि रोक्न कार्बन उत्सर्जन घटाउनुपर्छ।"),
    ("Prolonged drought affected drinking water supplies.",
     "लामो खडेरीले खानेपानी आपूर्तिमा असर पर्‍यो।"),
    ("Flooding displaced hundreds of households last year.",
     "गत वर्ष बाढीले सयौं घरपरिवार विस्थापित भए।"),
    # Tourism
    ("Pokhara is known for its scenic lakeside views.",
     "पोखरा आफ्नो मनोरम फेवाताल किनाराका लागि परिचित छ।"),
    ("Trekkers need a permit to enter the conservation area.",
     "संरक्षण क्षेत्र प्रवेश गर्न पदयात्रीलाई अनुमतिपत्र चाहिन्छ।"),
    ("The heritage site attracts thousands of visitors each year.",
     "सम्पदा स्थलले हरेक वर्ष हजारौं पर्यटक आकर्षित गर्छ।"),
    ("Several new hotels opened near the airport.",
     "विमानस्थल नजिक थुप्रै नयाँ होटल खुले।"),
    ("Homestay tourism has grown in the eastern hills.",
     "पूर्वी पहाडमा होमस्टे पर्यटन बढेको छ।"),
    # Admin
    ("The committee shall submit the report within 30 days.",
     "समितिले तीस दिनभित्र प्रतिवेदन पेस गर्नुपर्नेछ।"),
    ("Submit the application to the district office.",
     "निवेदन जिल्ला कार्यालयमा बुझाउनुहोस्।"),
    ("The ministry issued a public notice yesterday.",
     "मन्त्रालयले हिजो सार्वजनिक सूचना जारी गर्‍यो।"),
    ("Applicants must attach a copy of their citizenship.",
     "निवेदकले नागरिकताको प्रतिलिपि संलग्न गर्नुपर्छ।"),
    ("The department will publish the results next week.",
     "विभागले अर्को हप्ता नतिजा प्रकाशन गर्नेछ।"),
    # Health
    ("Wash hands frequently to prevent the spread of infection.",
     "संक्रमण फैलिनबाट रोक्न पटक पटक हात धुनुहोस्।"),
    ("The hospital reported a rise in dengue cases.",
     "अस्पतालले डेंगुका बिरामी बढेको जनाएको छ।"),
    ("Children should receive the vaccine on schedule.",
     "बालबालिकाले तोकिएको समयमा खोप लगाउनुपर्छ।"),
    ("A balanced diet reduces the risk of heart disease.",
     "सन्तुलित आहारले मुटुरोगको जोखिम घटाउँछ।"),
    ("Patients with symptoms should consult a doctor.",
     "लक्षण देखिएका बिरामीले चिकित्सकसँग परामर्श गर्नुपर्छ।"),
    # Law
    ("Every citizen has the right to a fair trial.",
     "प्रत्येक नागरिकलाई निष्पक्ष सुनुवाइको अधिकार छ।"),
    ("The court dismissed the petition for lack of evidence.",
     "प्रमाण नपुगेकाले अदालतले निवेदन खारेज गर्‍यो।"),
    ("This clause shall come into force immediately.",
     "यो दफा तुरुन्त लागू हुनेछ।"),
    ("The tribunal will hear the appeal next month.",
     "न्यायाधिकरणले अर्को महिना पुनरावेदन सुनुवाइ गर्नेछ।"),
    ("Violation of the act carries a monetary penalty.",
     "ऐन उल्लङ्घन गरेमा आर्थिक जरिवाना हुनेछ।"),
    # Education
    ("The textbook covers the fundamentals of arithmetic.",
     "पाठ्यपुस्तकले अंकगणितका आधारभूत कुरा समेट्छ।"),
    ("Students must register before the examination begins.",
     "परीक्षा सुरु हुनुअघि विद्यार्थीले दर्ता गर्नुपर्छ।"),
    ("The school revised its curriculum this year.",
     "विद्यालयले यस वर्ष पाठ्यक्रम परिमार्जन गर्‍यो।"),
    ("Teachers attended a week-long training programme.",
     "शिक्षकहरूले एक हप्ते तालिम कार्यक्रममा भाग लिए।"),
    ("Scholarships are available for students from remote areas.",
     "दुर्गम क्षेत्रका विद्यार्थीका लागि छात्रवृत्ति उपलब्ध छ।"),
    # General
    ("The event was attended by hundreds of people.",
     "कार्यक्रममा सयौं मानिसको उपस्थिति थियो।"),
    ("He did not say anything about the matter.",
     "उनले यस विषयमा केही भनेनन्।"),
    ("The match ended in a draw on Sunday evening.",
     "आइतबार साँझ खेल बराबरीमा टुंगियो।"),
    ("She bought a new pair of shoes from the market.",
     "उनले बजारबाट नयाँ जुत्ता किनिन्।"),
    ("The meeting has been postponed until further notice.",
     "अर्को सूचना नआएसम्म बैठक स्थगित गरिएको छ।"),
]


def main():
    rows = [{"id": f"SAMPLE_{i + 1:04d}", "english": e, "nepali": n}
            for i, (e, n) in enumerate(PAIRS)]
    df = pd.DataFrame(rows)

    half = len(df) // 2

    # workbook 1: ID / src / tgt
    a = df.iloc[:half].copy()
    a.columns = ["ID", "src", "tgt"]
    a.to_excel(os.path.join(HERE, "sample_part_I.xlsx"), index=False)

    # workbook 2: ID / SOURCE / TARGET - same data, different headers, which
    # is exactly how the real source files differ
    b = df.iloc[half:].copy()
    b.columns = ["ID", "SOURCE", "TARGET"]
    b.to_excel(os.path.join(HERE, "sample_part_II.xlsx"), index=False)

    print(f"wrote 2 sample workbooks to {HERE} ({len(df)} pairs total)")
    print("try it:")
    print("  set NLTM_DATA_DIR=<a scratch dir>")
    print("  copy sample_data\\*.xlsx <scratch>\\data_raw\\")
    print("  python merge_data.py")


if __name__ == "__main__":
    main()
