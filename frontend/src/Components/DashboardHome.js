import React, { useState } from 'react'
import Welcome from './Welcome'
import QuickSummary from './QuickSummary'
import Pie from './graph'
import UserData from './Dashboardpiedata'
import Table from './table'

const DashboardHomePage = (props) => {
  const [userData, setUserData] = useState({
    labels: UserData.map((data) => data.year),
    datasets: [
      {
        label: "Users Gained",
        data: UserData.map((data) => data.userGain),
        backgroundColor: [
          "rgba(75,192,192,1)",
          "#ecf0f1",
          "#50AF95",
          "#f3ba2f",
          "#2a71d0",
        ],
        borderColor: "black",
        borderWidth: 2,
      },
    ],
  });
  return (
    <div>
        <div className='dashboard-header'>
          <Welcome nclose={props.nclose1}/>
        </div>
        <div className='dashboard-body'>
          <QuickSummary/>
          <div className="piechart" style={{ width: 500 }}>
        <Pie chartData={userData} />
        </div>
        <div className='dashtable'>
          <Table />
        </div>
        </div>


        
    </div>
  )
  }

export default DashboardHomePage