import React from 'react'
import MainHeader from '../Components/MainHeader'
import image from "../Pages/Customer_Care.png";
import { motion } from 'framer-motion'

const FAQ = () => {
  return (
    <motion.div animate={{width:"100%"}} initial={{width:0}} exit={{x:window.innerWidth,transition:{duration:1}}}>
      <div className='Faq'>
      <MainHeader />
      <h1>Frequently Asked Questions</h1>


      <div className="row featurette">
      <div className="col-md-7">
        <h2 className="featurette-heading fw-normal lh-1">How do I get started with Cosmic Ledger? </h2>
        <p className="lead">Creating an account with Cosmic Ledger is simple and straightforward. Simply visit our website and click on the "Sign Up" button. You can create an account using your email address. Once you've created an account, you can start adding your financial accounts and tracking your transactions.  </p>
      </div>
      <div className="col-md-5">
      <img 
     src="https://images.unsplash.com/photo-1499750310107-5fef28a66643?q=80&w=2370&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
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
        <h2 className="featurette-heading fw-normal lh-1">How do I use Cosmic Ledger to track my spending?</h2>
        <p className="lead">Cosmic Ledger provides a variety of tools to help you track your spending. You can view your spending by category, merchant, or date. You can also set spending goals and track your progress towards them.</p>
      </div>
      <div className="col-md-5 order-md-1">
      <img 
     src="https://plus.unsplash.com/premium_photo-1676673222100-203637d00d94?q=80&w=2500&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" 
     className="featurette-image img-fluid mx-auto" 
     alt="Responsive image" 
     width="500" 
     height="500" 
     preserveAspectRatio="xMidYMid slice" 
     focusable="false"
    />
      </div>
      <hr className="featurette-divider"></hr>


      
    </div>
    <hr className="featurette-divider"></hr>


    <div className="row featurette">
      <div className="col-md-7">
        <h2 className="featurette-heading fw-normal lh-1">How do I contact Cosmic Ledger customer support? </h2>
        <p className="lead">You can contact Cosmic Ledger customer support by email or live chat. Our customer support team is available Monday through Friday from 9:00 AM to 5:00 PM IST.</p>
      </div>
      <div className="col-md-5">
      <img 
    src={image}
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

      
             
    

              
    </motion.div>
  )
}

export default FAQ