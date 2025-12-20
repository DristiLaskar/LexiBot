// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Auction {
    address public owner;
    address public highestBidder;
    uint public highestBid;

    constructor() {
        owner = msg.sender;
    }

    function bid() external payable {
        require(msg.value > highestBid, "Bid too low");
        if (highestBidder != address(0)) {
            payable(highestBidder).transfer(highestBid); // refund
        }
        highestBidder = msg.sender;
        highestBid = msg.value;
    }

    function closeAuction() external {
        require(msg.sender == owner, "Only owner");
        payable(owner).transfer(address(this).balance);
    }
}
