import React from 'react'
import MainHeader from '../Components/MainHeader'
import { motion } from 'framer-motion'
import image1 from "../Pages/logo_pic.jpg"
import image2 from "../Pages/About_us_meticulous_planning.png"
import image3 from "../Pages/About_us_financial_clarity.png"
import image4 from "../Pages/About_us_empowered_decision.png"
import image5 from "../Pages/About_Us_Financialharmony.png"
const About_Us = () => {
  return (
    <motion.div className='aboutUsPage' animate={{width:"100%"}} initial={{width:0}} exit={{x:window.innerWidth,transition:{duration:1}}}>
      <div className='About_UsFull'>
      <MainHeader />
    <div container="About_Uscont" className='About_Uscont'>
       <div container="Heading_About_Us " className='Heading_About_Us'>
        <h1 >About Us</h1>

        </div>

      
        <div className="row featurette">
      <div className="col-md-7">
        <h2 className="featurette-heading fw-normal lh-1">Introducing Cosmic Ledger: Your Personal Financial Guardian </h2>
        <p className="lead">In the realm of personal finance, Cosmic Ledger emerges as your trusted companion, meticulously tracking your financial transactions and guiding you through the complexities of your financial cosmos. Inspired by the Egyptian god Thoth, the deity of knowledge, wisdom, and writing, Cosmic Ledger embodies the essence of financial clarity and organization.</p>
        <p className="lead">Just as Thoth inscribed every action and decision upon the celestial tablets, Cosmic Ledger meticulously chronicles your financial journey, ensuring that every penny is accounted for and every decision is made with informed precision. Whether you're navigating the intricacies of personal finances or managing the complexities of business ventures, Cosmic Ledger stands as your financial guardian, ensuring that your financial path is illuminated with the brilliance of financial literacy and empowered decision-making.</p>
      </div>
      <div className="col-md-5">
      <img 
     src={image1} 
     className="featurette-image img-fluid mx-auto" 
     alt="Responsive image" 
     width="500" 
     height="500" 
     preserveAspectRatio="xMidYMid slice" 
     focusable="false"
    />
      </div>
    </div>
    <hr className="featurette-divider"></hr>


    <div className="row featurette">
      <div className="col-md-7 order-md-2">
        <h2 className="featurette-heading fw-normal lh-1">Your Meticulously Organized Financial Journey </h2>
        <p className="lead">Imagine your financial transactions as celestial bodies, each charting a unique course through the vast expanse of your financial cosmos. Cosmic Ledger meticulously maps these transactions, creating a comprehensive constellation of your financial activities. With every transaction recorded, Cosmic Ledger provides a clear and organized view of your financial landscape, enabling you to make informed decisions that align with your financial goals.</p>
      </div>
      <div className="col-md-5 order-md-1">
      <img 
     src={image2} 
     className="featurette-image img-fluid mx-auto" 
     alt="Responsive image" 
     width="500" 
     height="500" 
     preserveAspectRatio="xMidYMid slice" 
     focusable="false"
    />
      </div>
    </div>
    <hr className="featurette-divider"></hr>


    <div className="row featurette">
      <div className="col-md-7">
        <h2 className="featurette-heading fw-normal lh-1">Financial Clarity illuminated by Knowledge</h2>
        <p className="lead">Just as Thoth enlightened the gods with his wisdom, Cosmic Ledger empowers you with financial clarity. Through its comprehensive reporting and intuitive interface, Cosmic Ledger transforms your financial data into actionable insights, helping you identify patterns, understand trends, and make informed decisions that align with your financial aspirations.</p>
      </div>
      <div className="col-md-5">
      <img 
     src={image3}
     className="featurette-image img-fluid mx-auto" 
     alt="Responsive image" 
     width="500" 
     height="500" 
     preserveAspectRatio="xMidYMid slice" 
     focusable="false"
    />
      </div>
    </div>
    <hr className="featurette-divider"></hr>   



    <div className="row featurette">
      <div className="col-md-7 order-md-2">
        <h2 className="featurette-heading fw-normal lh-1">Empowered Decision-Making, Guided by Wisdom</h2>
        <p className="lead">Cosmic Ledger is more than just a financial tracker; it's a personal financial advisor, guiding you towards financial well-being. With it's advanced analytics, Cosmic Ledger helps you anticipate future financial needs, plan for upcoming expenses, and make informed investment decisions.</p>
      </div>
      <div className="col-md-5 order-md-1">
      <img 
     src={image4} 
     className="featurette-image img-fluid mx-auto" 
     alt="Responsive image" 
     width="500" 
     height="500" 
     preserveAspectRatio="xMidYMid slice" 
     focusable="false"
    />
      </div>
    </div>
    <hr className="featurette-divider"></hr>



    <div className="row featurette">
      <div className="col-md-7">
        <h2 className="featurette-heading fw-normal lh-1">Embrace Financial Harmony with Cosmic Ledger</h2>
        <p className="lead">In the ever-evolving realm of finance, Cosmic Ledger stands as your unwavering companion, harmonizing your financial transactions and guiding you towards a brighter financial future. Just as Thoth orchestrated the celestial movements, Cosmic Ledger orchestrates your financial journey, ensuring that every decision is made with clarity, wisdom, and empowerment.</p>
      </div>
      <div className="col-md-5">
      <img 
     src={image5}
     className="featurette-image img-fluid mx-auto" 
     alt="Responsive image" 
     width="500" 
     height="500" 
     preserveAspectRatio="xMidYMid slice" 
     focusable="false"
    />
      </div>
    </div>
    <hr className="featurette-divider"></hr>   

        
    </div>
    </div>
    </motion.div>
  )
}

export default About_Us