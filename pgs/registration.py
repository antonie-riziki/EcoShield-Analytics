import streamlit as st
import africastalking
import os
import sys



sys.path.insert(1, './modules')

from func import welcome_message
from dotenv import load_dotenv

load_dotenv()



col1, col2 = st.columns(2)

with col1:
	with st.form(key="user_registration"):
	    st.subheader("Registration")
	    fname, sname = st.columns(2)
	    with fname:
	    	first_name = st.text_input("First Name")
	    with sname:
	    	surname = st.text_input("Surname")
	    
	    username = st.text_input('Username:')
	    email = st.text_input("Email: ")
	    phone_number = st.number_input("Phone Number:", value=None, min_value=0, max_value=int(10e10))
	    password = st.text_input('Passowrd', type="password")
	    confirm_password = st.text_input('Confirm password', type='password')
	    face_id = st.file_uploader('Profile Photo')

	    checkbox_val = st.checkbox("Subscribe to our Newsletter")

	    submit_personal_details = st.form_submit_button("Submit", use_container_width=True, type="primary")

	    
	    if password != confirm_password:
	    	st.error('Password mismatch', icon='⚠️')

	    else:
		    
		    if not (email and password):
		    	st.warning('Please enter your credentials!', icon='⚠️')
		    	# st.toast('You Must Register your Personal Details', icon='⚠️')
		    else:
		    	st.success('Proceed to engaging with the system!', icon='👉')

		    	

		    	if submit_personal_details:
		    		welcome_message(first_name, phone_number)


with col2:
	st.image('https://images.squarespace-cdn.com/content/v1/6049e33a3512a120620cfe14/1620890817516-PQCWJG5OQPRN6D8BD83G/1000_1.jpg', width=700)
	st.image('https://preview.redd.it/mean-august-temperature-by-region-v0-psgue16vvgxe1.png?width=640&crop=smart&auto=webp&s=ecdb13800198e71089e811c313f440f5ffbc2fb3', width=800)
	st.image('https://www.weather2visit.com/public/map/nakasongola-ug.png', width=900)