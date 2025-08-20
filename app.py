import streamlit as st 



reg_page = st.Page("./pgs/registration.py", title="register", icon=":material/person_add:")
# signin_page = st.Page("./pgs/signin.py", title="sign in", icon=":material/login:")
home_page = st.Page("./pgs/main.py", title="Getting Started", icon=":material/home:")
dashboard_page = st.Page("./pgs/dashboard.py", title="Dashboard", icon=":material/app_registration:")
automation_page = st.Page("./pgs/automation.py", title="Live Automation", icon=":material/apk_install:")
# sms_service_page = st.Page("./pgs/sms_service.py", title="Messaging", icon=":material/sms:")
# airtime_page = st.Page("./pgs/airtime.py", title="Airtime", icon=":material/redeem:")
# mobile_data_page = st.Page("./pgs/mobile_data.py", title="Mobile Data", icon=":material/lte_plus_mobiledata_badge:")
# ussd_page = st.Page("./pgs/ussd.py", title="USSD", icon=":material/linked_services:")
chatbot_page = st.Page("./pgs/chatbot.py", title="Ask EcoShield", icon=":material/chat:")







pg = st.navigation([reg_page, home_page, dashboard_page, automation_page, chatbot_page])

st.set_page_config(
    page_title="EcoShield Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.echominds.africa',
        'Report a bug': "https://www.echominds.africa",
        'About': """
            EcoShield Analytics is an interactive dashboard that visualizes community-level vulnerability by combining climate exposure, food security, and 
            nutrition adequacy data. It helps policymakers, NGOs, and planners anticipate risks, design targeted interventions, and 
            strengthen community resilience
        """
    }
)

pg.run()

