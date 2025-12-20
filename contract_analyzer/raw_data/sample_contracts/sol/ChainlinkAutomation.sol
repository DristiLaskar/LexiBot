// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {StreamsLookupCompatibleInterface} from "@chainlink/contracts/src/v0.8/automation/interfaces/StreamsLookupCompatibleInterface.sol";
import {ILogAutomation, Log} from "@chainlink/contracts/src/v0.8/automation/interfaces/ILogAutomation.sol";
import {IVerifierProxy} from "@chainlink/contracts/src/v0.8/llo-feeds/interfaces/IVerifierProxy.sol";
import {IVerifierFeeManager} from "@chainlink/contracts/src/v0.8/llo-feeds/interfaces/IVerifierFeeManager.sol";
import {IRewardManager} from "@chainlink/contracts/src/v0.8/llo-feeds/interfaces/IRewardManager.sol";
import {Common} from "@chainlink/contracts/src/v0.8/llo-feeds/libraries/Common.sol";
import {IERC20} from "@chainlink/contracts/src/v0.8/vendor/openzeppelin-solidity/v4.8.3/contracts/interfaces/IERC20.sol";

contract DataStreamsAutomation is ILogAutomation, StreamsLookupCompatibleInterface {
    error InvalidReportVersion(uint16 version);

    IVerifierProxy public verifier;
    string public constant FEED_LABEL = "feedIDs";
    string public constant QUERY_LABEL = "timestamp";
    string[] public feedIds;

    int192 public latestPrice;

    constructor(address _verifier, string[] memory _feedIds) {
        verifier = IVerifierProxy(_verifier);
        feedIds = _feedIds;
    }

    function checkLog(Log calldata log, bytes memory) external returns (bool upkeepNeeded, bytes memory performData) {
        revert StreamsLookup(FEED_LABEL, feedIds, QUERY_LABEL, log.timestamp, "");
    }

    function checkCallback(bytes[] calldata values, bytes calldata extraData) external pure returns (bool, bytes memory) {
        return (true, abi.encode(values, extraData));
    }

    function performUpkeep(bytes calldata performData) external {
        (bytes[] memory signedReports, bytes memory extraData) = abi.decode(performData, (bytes[], bytes));
        bytes memory unverifiedReport = signedReports[0];

        (, bytes memory reportData) = abi.decode(unverifiedReport, (bytes32[3], bytes));
        uint16 reportVersion = (uint16(uint8(reportData[0])) << 8) | uint16(uint8(reportData[1]));

        if (reportVersion != 3 && reportVersion != 4) {
            revert InvalidReportVersion(reportVersion);
        }

        IVerifierFeeManager feeManager = verifier.s_feeManager();
        address feeToken = feeManager.i_linkAddress();
        (Common.Asset memory fee,,) = feeManager.getFeeAndReward(address(this), reportData, feeToken);

        IERC20(feeToken).approve(feeManager.i_rewardManager(), fee.amount);
        bytes memory verifiedData = verifier.verify(unverifiedReport, abi.encode(feeToken));

        if (reportVersion == 3) {
            (bytes32 feedId, uint32 validFromTimestamp, uint32 observationsTimestamp, uint192 nativeFee, uint192 linkFee, uint32 expiresAt, int192 price, int192 bid, int192 ask) = abi.decode(verifiedData, (bytes32, uint32, uint32, uint192, uint192, uint32, int192, int192, int192));
            latestPrice = price;
        } else {
            (bytes32 feedId, uint32 validFromTimestamp, uint32 observationsTimestamp, uint192 nativeFee, uint192 linkFee, uint32 expiresAt, int192 price, uint32 marketStatus) = abi.decode(verifiedData, (bytes32, uint32, uint32, uint192, uint192, uint32, int192, uint32));
            latestPrice = price;
        }
    }
}
