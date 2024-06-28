import { useState } from "react";
import React from "react";
import NavBar from "../Components/NavBar";
import Transactionbody from "../Components/transactionbody";

const TransactionHistory=()=>{
  const [togg,setTogg]=useState(true);
  const toggnav = () => {
    setTogg((prevTogg) => !prevTogg);
  };
    return (
        <div>
            <div className='section-20'>
        {togg && <NavBar />}
        </div>
            <div className='section-80'>
            <Transactionbody nclose1={toggnav}/>
        </div>
        </div>
      )
}

export default TransactionHistory