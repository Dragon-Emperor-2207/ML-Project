import './App.css';
import Home from './Pages/Home';
import AboutUs from './Pages/About_Us';
import Pricing from './Pages/Pricing';
import Dashboard from './Pages/Dashboard';
import FAQ from './Pages/FAQ'
import TransactionHistory from './Pages/TransactionHistory';
import AnimatedRoutes from './Components/AnimatedRoutes';


import {
  BrowserRouter as Router,
  Route,
  Routes
} from 'react-router-dom'
import React from 'react'
import MainHeader from './Components/MainHeader';

function App() {
  return (
    
    <div className="App" >
        <Router>
        {/* <Route path='/login/:loginid' element = {<Dashboard />}/>
        <Route path='/login/:loginid/transactions' element = {<TransactionHistory />}/>
        <Route path='/home' element = {<Home/>}/>
        <Route path='/about' element = {<AboutUs/>}/>
        <Route path='/pricing' element = {<Pricing/>}/>
        <Route path='/faq' element = {<FAQ/>}/> */}
        <AnimatedRoutes />
      </Router>
    </div>
  );
}

export default App;
