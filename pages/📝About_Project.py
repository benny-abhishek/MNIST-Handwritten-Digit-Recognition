import streamlit as st 

st.markdown("""
<html>
  <head>
    <title>About Us - MNIST Handwritten Digit Recognition</title>
    <style>
      /* Style the header */
      header {
        background-color: #000000;
        padding: 20px;
        text-align: center;
      }
      /* Style the container */
      .container {
        margin: 50px;
      }
      /* Style the section */
      section {
        margin: 20px;
        padding: 20px;
      }
      /* Style the footer */
      footer {
        background-color: #000000;
        padding: 20px;
        text-align: center;
      }
    </style>
  </head>
  <body>
    <header>
      <h1>About Us</h1>
    </header>
    <div class="container">
      <section>
        <h2>About</h2>
        <p>
          The prediction of MNIST Handwritten Digits Datatset has been done using ML Algorithms
            like SVM, KNN, Logistic Regression and Multi-layer Perceptron.
            Different kinds of transformations has been done and the algorithms have
            been applied for the transformed data. One-Way ANOVA has been done
            for comparison of algorithms before and after transformation. Also includes a 
            feature for the user to draw a digit which will be predicted. 
        </p>
      </section>
      <section>
        <h2>Our Team</h2>
        <li>Benny Abhishek - 21PD03</li>
        <li>Aayush Srivatsav - 21PD25</li>
      
  <h2>Libraries Used</h2>
  <ul>
    <li>Pandas</li>
    <li>Numpy</li>
    <li>Matplotlib</li>
    <li>Streamlit</li>
    <li>Keras</li>
    <li>Gradio</li>
  </ul>
  <h2>Methods Used</h2>
  <ul>
    <li>SVM</li>
    <li>Logistic Regression</li>
    <li>KNN</li>
    <li>Multi Layer Perceptron</li>
  </ul>
</section>
      <section>
  <h2>References</h2>
  <h3>Streamlit</h3>
  <ul>
    <li>https://streamlit.io/</li>
  </ul>
  <ul>
    <footer>
      <p>&copy; MNIST Handwritten Digit Recognition</p>
    </footer>
  </body>
</html>""",unsafe_allow_html=True)