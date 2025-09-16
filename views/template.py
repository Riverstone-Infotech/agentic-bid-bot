import base64
import requests
import mimetypes
import json
from jinja2 import Template
from datetime import datetime, date
import os
    
def render_quotation(progress: dict,today) -> str:
    total_amount=0
    for item in progress.get("furniture_items_and_pricing"):
        # item["total amount"] = item.get("quantity") * item.get("unit price")
        total_amount+=float(item.get("total amount"))
        # split description into parts
        parts = [p.strip() for p in item["description"].replace(",", "|").split("|") if p.strip()]
        item["reference"] = parts[0]
        item["attributes"] = parts[1:]

    date_obj = datetime.strptime(progress["Quotation Details"]["Issue date"], "%B %d, %Y")

    # Get the day name
    day_name = date_obj.strftime("%A").upper()
    progress["Quotation Details"]["Issue day"] = day_name
    progress["totals"] = {
    "subtotal": total_amount,
    "grand_total": total_amount
}
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RFQ #023-2013 - Furniture for Dan Kinney and Chesterfield Family Centers</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.4;
                color: #333;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .rfq-number {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .section {
                margin-bottom: 25px;
            }
            .section-title {
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 15px;
                color: #2c5aa0;
                border-bottom: 2px solid #2c5aa0;
                padding-bottom: 5px;
            }
            .contact-info {
                background-color: #f5f5f5;
                padding: 15px;
                border-left: 4px solid #2c5aa0;
                margin: 15px 0;
                display: flex;
                justify-content: space-around;
            }
            .requirements-list {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
            }
            .requirements-list ul {
                margin: 0;
                padding-left: 20px;
            }
            .important-notice {
                background-color: #fffbcc;
                border: 1px solid #f0c040;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .quotation-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .quotation-table th,
            .quotation-table td {
                border: 1px solid #333;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }
            .quotation-table th {
                background-color: #2c5aa0;
                color: white;
                font-weight: bold;
            }
            /* REMOVE automatic header repetition */
            @media print {
                .quotation-table thead {
                    display: table-row-group !important;
                }
                .quotation-table tr {
                    page-break-inside: avoid !important;
                }
            }
            .group-header {
                background-color: #e6f2ff;
                font-weight: bold;
            }
            .specifications-item {
                background-color: #f9f9f9;
                margin: 10px 0;
                padding: 10px;
                border-left: 3px solid #2c5aa0;
            }
            .quotation-table thead {
                display: table-header-group;
            }
            .form-section {
                background-color: #f5f5f5;
                padding: 15px;
                margin: 20px 0;
                border: 2px solid #2c5aa0;
            }
            .checkbox-list {
                list-style-type: none;
                padding-left: 0;
            }
            .checkbox-list li {
                margin: 10px 0;
            }
            .signature-line {
                border-bottom: 1px solid #333;
                display: inline-block;
                min-width: 200px;
                margin: 0 10px;
            }
            .page-break {
                border-top: 2px dashed #ccc;
                margin: 30px 0;
                padding-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{progress["Client Information"]["company name"]}}</h1>
            <h2>REQUEST FOR QUOTATION</h2>
            <div class="rfq-number">RFQ No: {{progress["Quotation Details"]["Quotation ID"]}}</div>
            <p><strong>THIS IS NOT AN ORDER</strong></p>
        </div>

        <div class="contact-info">
            <p><strong>TO:</strong> {{progress["Client Information"]["Name"]}}<br>
            {{progress["Client Information"]["Company"].replace(',','<br>')}}<br>
            {{''.join(progress["Client Information"]["Address"].split(',')[:-2])}}<br>
            {{''.join(progress["Client Information"]["Address"].split(',')[-2:])}}</p>
            
            <p><strong>Date Issued:</strong> {{progress["Quotation Details"]["Issue date"]}}<br>
            <strong>Buyer's Email:</strong> {{progress["Client Information"]["email"]}} <br>
            <strong>Telephone Number:</strong> 417-864-1621<br>
            <strong>DUE DATE:</strong> {{progress["Quotation Details"]["Due Date"]}}</p>
        </div>

        <div class="important-notice">
            <p><strong>QUOTATIONS MUST BE PHYSICALLY RECEIVED IN THE DIVISION OF PURCHASES PRIOR TO {{progress["Quotation Details"]["Due Time"]}} ON {{progress["Quotation Details"]["Issue day"]}}, {{progress["Quotation Details"]["Due Date"]}}.</strong></p>
        </div>

        <div class="requirements-list">
            <ul>
                <li>Quotations shall be submitted on the forms provided and must be manually signed.</li>
                <li>Quotations shall be submitted with the RFQ number clearly indicated.</li>
                <li>Quotations and all required documentation may be faxed to Fax #417-864-1927.</li>
                <li>Quotations received after the bid opening date and time shall be rejected.</li>
                <li>The attached Terms and Conditions shall become part of any purchase order resulting from this RFQ.</li>
            </ul>
        </div>

        <div class="section">
            <div class="section-title">DESCRIPTION</div>
            <h3>FURNITURE FOR {{progress["Client Information"]["Company"]}}</h3>
            <p>See attached General Conditions, Specifications, and Quotation Form for detailed information.</p>
            
            <p><strong>DELIVERY:</strong> F.O.B. DESTINATION<br>
            The articles to be furnished hereunder shall be delivered all transportation charges paid by the bidder to destination.</p>
        </div>

        <div class="page-break">
            <div class="section-title">INSTRUCTION TO BIDDERS</div>
            
            <div class="section">
                <h4>01. Opening Location</h4>
                <p>The Quotations will be opened at the {{progress['Client Information']['Company']}}, {{progress['Client Information']['Address']}} in the presence of Purchasing officials at the due date and time indicated on the RFQ. All bidders or their representatives are invited to attend the opening of the RFQ.</p>
            </div>

            <div class="section">
                <h4>02. RFQ Delivery Requirements</h4>
                <p>Any Quotations received after the above stated time and date will not be considered. It shall be the sole responsibility of the bidder to have their Quotation delivered to the Division of Purchases for receipt on or before the due date and time indicated. If a Quotation is sent by U.S. Mail, the bidder shall be responsible for its timely delivery to the Division of Purchases office. Quotations delayed by mail shall not be considered, shall not be opened, and shall be rejected. Arrangements may be made for their return at the bidder's request and expense. Quotations may be faxed to the Division of Purchases and accepted if the signed quotation form and required information is faxed and received prior to the due date and time. Quotations sent by email will not be accepted.</p>
            </div>

            <div class="section">
                <h4>03. Sealed and Marked</h4>
                <p>If sent by mail, one original signed Quotation shall be submitted in one sealed package, clearly marked on the outside of the package with the Request for Quotation number and addressed to:</p>
                <div style="display: block" class="contact-info">
                    {{progress["Client Information"]["company name"]}}<br>
                    Division of Purchases<br>
                    {{''.join(progress["Client Information"]["Address"].split(',')[:-2])}}<br>
                    {{''.join(progress["Client Information"]["Address"].split(',')[-2:])}}
                </div>
            </div>

            <div class="section">
                <h4>04. Legal Name and Signature</h4>
                <p>Quotations shall clearly indicate the legal name, address, and telephone number of the bidder (company, firm, corporation, partnership, or individual). Quotations shall be manually signed above the printed name and title of signer on the Affidavit of Compliance page. The signer shall have the authority to bind the company to the submitted Quotation. Failure to properly sign the Quote form shall invalidate same, and it shall not be considered for award.</p>
            </div>
        </div>

        <div class="page-break">
            <div class="section-title">GENERAL TERMS AND CONDITIONS</div>
            
            <div class="section">
                <h4>1. PURPOSE:</h4>
                <p>These specifications establish the minimum requirements for furniture to be used {{progress["Client Information"]["Company"]}}.</p>
            </div>

            <div class="section">
                <h4>2. LOCATION:</h4>
                <p>{{progress["Client Information"]["company name"]}} {{progress["Client Information"]["Address"]}}.</p>
            </div>

            <div class="section">
                <h4>3. QUANTITIES:</h4>
                <p>Quantities listed are estimates only and may be more or less based on prices submitted and available funds.</p>
            </div>

            <div class="section">
                <h4>4. AWARD:</h4>
                <p>In making an award the City will look at price, delivery, and warranty. Delivery is desired no later than October 8, 2012 so furniture will be available for the Dan Kinney Family Center grand opening. The City reserves the right to make separate awards for each line, group, or combination thereof.</p>
            </div>

            <div class="section">
                <h4>5. SCOPE OF WORK:</h4>
                <p>a. Provide: Furniture, deliver, the Contractor shall be responsible for all installation, if required and removal and disposal of all residual packing or shipping material.</p>
            </div>

            <div class="section">
                <h4>6. NEW PRODUCT:</h4>
                <p>All products supplied hereunder shall be new and the manufacturers standard model in current production. The product shall not be rebuilt, reconditioned, or refurbished. All products supplied hereunder shall, except as specified herein, fully conform to each and every specification, drawing, sample or other description, which is furnished to the City by the manufacturer and/or the Contractor.</p>
            </div>
        </div>

        <div class="page-break">
            <div class="section-title">SPECIFICATIONS</div>
            
            <div class="section">
                {% for item in progress.get("furniture_items_and_pricing") %}
                <div class="specifications-item">
                    <h4>{{ item.reference }}</h4>
                    <p><strong>Reference:</strong> {{ item.reference }}</p>
                    <ul>
                        {% for attr in item.attributes %}
                        <li>{{ attr }}</li>
                        {% endfor %}
                        <li><strong>Quantity: {{ item.quantity }}</strong></li>
                    </ul>
                </div>
                {% endfor %}

            </div>
        </div>

        <div class="page-break">
            <div class="section-title">QUOTATION FORM - PROPOSAL</div>
            
            <div class="form-section">
                <p><strong>SUBMITTED BY:</strong> {{progress["Enterprise Information"]["name"]}} </p>
                
                <p>Pursuant to and in accordance with the above stated Request for Quotation, the undersigned hereby declares that they have examined the RFQ documents and specifications for the item(s) listed below.</p>
                
                <p>The undersigned proposes and agrees, if their Bid is accepted to furnish the item(s) submitted below, including delivery to Springfield, Missouri in accordance with the delivery schedule indicated below and according to the prices products/services information submitted.</p>
            </div>

            <table class="quotation-table">
                <thead>
                    <tr>
                        <th>ITEM</th>
                        <th>PRODUCT CODE</th>
                        <th>DESCRIPTION</th>
                        <th>QTY</th>
                        <th>UNIT PRICE</th>
                        <th>TOTAL AMOUNT</th>
                    </tr>
                </thead>
                <tbody>
                {% for item in progress.get("furniture_items_and_pricing") %}
                    <tr>
                        <td>{{ loop.index }}.</td>
                        <td>{{ item["product code"] }}
                        <td>
                            {{ item.reference }}{% if item.attributes %}, {{ item.attributes | join(', ') }}{% endif %}.<br>
                        </td>
                        <td>{{ item.quantity }}</td>
                        <td>$ {{ "%.2f"|format(item['unit price']) }}</td>
                        <td>$ {{ "%.2f"|format(item['total amount']) }}</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>

            <div class="form-section">
                <p><strong>DELIVERY:</strong> F.O.B. DESTINATION</p>
                <p><strong>ACCEPT VISA P-CARD:</strong> YES________ NO________</p>
                <p>Prompt Payment Discount __________% _________ Days, Net _____ Days</p>
            </div>
        </div>

        <div class="page-break">
            <div class="section-title">AFFIDAVIT OF COMPLIANCE</div>
            
            <div class="form-section">
                <p><strong>To be submitted with Vendor's Quotation</strong></p>
                
                <p>_____We DO NOT take exception to the RFQ Documents/Requirements.</p>
                <p>_____We TAKE exception to the RFQ Documents/Requirements as follows:</p>
                
                <p><strong>Specific exceptions are as follows:</strong></p>
                <div style="min-height: 100px; border: 1px solid #ccc; padding: 10px; margin: 10px 0;"></div>
                
                <p>I have carefully examined the Request for Quotation and agree to abide by all submitted pricing, delivery, terms and conditions of this Quotation unless otherwise stipulated herein.</p>
                
                <table style="width: 100%; margin-top: 20px;">
                    <tr>
                        <td style="border: none; padding: 10px;">
                            <strong>Company Name:</strong> {{progress["Enterprise Information"]["name"]}}<br><br>
                            <strong>By:</strong> <span class="signature-line"></span><br>
                            <span style="margin-left: 40px;">{{progress['Enterprise Information']['contactName']}}</span><br><br>
                            
                        </td>
                        <td style="border: none; padding: 10px;">
                            <strong>Company Address:</strong><br>
                            {{progress["Enterprise Information"]["address"]}}<br><br>
                            <strong>Telephone Number:</strong> {{progress["Enterprise Information"]["phoneNumber"]}}<br><br>
                            <strong>Email:</strong> {{progress["Enterprise Information"]["email"]}}<br><br>
                            <strong>Date:</strong> {{today}}
                        </td>
                    </tr>
                </table>
                
                <div style="margin-top: 30px;">
                    <p><strong>ADDENDA</strong></p>
                    <p>Bidder acknowledges receipt of the following addendum:</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                </div>
            </div>
        </div>


        <footer style="margin-top: 50px; text-align: center; font-size: 12px; color: #666;">
            <p>{{progress["Client Information"]["company name"]}}<br>
            {{progress["Client Information"]["Address"]}}<br>
            {% if progress["Client Information"]["phone"] %}Phone: {{progress["Client Information"]["phone"]}} {% endif %} | {% if progress["Client Information"]["fax"] %}Fax: {{progress["Client Information"]["fax"]}} {% endif %}</p>
        </footer>

    </body>
    </html>
    """

    template = Template(html_template)
    return template.render(progress=progress,today = today)

def image_url_to_base64(url: str) -> str:
    """
    Downloads an image from a URL and returns a Base64 data URI.
    """
    response = requests.get(url)
    response.raise_for_status()

    mime_type = response.headers.get("Content-Type")
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(url)
        if not mime_type:
            mime_type = "image/jpeg"

    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def render_proposal(progress: dict,today,basic) -> str:
    sub_total = 0
    for code,data in progress.items():
        total_amount=0
        for item in data.get("furniture_items_and_pricing"):
            # item["total amount"] = item.get("quantity") * item.get("unit price")
            total_amount+=float(item.get("total amount"))
            # split description into parts
            parts = [p.strip() for p in item["RFP_description"].replace(",", "|").split("|") if p.strip()]
            item["reference"] = parts[0]
            item["attributes"] = parts[1:]

        date_obj = datetime.strptime(data["Quotation Details"]["Issue date"], "%B %d, %Y")

        # Get the day name
        day_name = date_obj.strftime("%A").upper()
        data["Quotation Details"]["Issue day"] = day_name
#         data["Dealer Information"] = {
#     "name": "ABC DEF",
#     "company": "Alphabet incorp",
#     "address": "1234 56st, GHI avenue, JKLMN OPQRS 567890",
#     "phone": "1029384756",
#     "email": "ABDE@contact.com",
#     "post":"purchse dealer",
#     "amount":0
#   }   
        path = os.path.join(os.path.dirname(__file__), "dealer.json")

        with open(path,'r') as f:
            data["Dealer Information"] = json.load(f)
        data["totals"] = {
        "subtotal": total_amount,
        "grand_total": total_amount
    }
        sub_total+=total_amount
    basic["sub_total"]=sub_total
    html_template = """
    <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IOM Furniture Proposal Response Template</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            color: #333;
        }

        .subsection-title {
            font-size: 16px;
            font-weight: bold;
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .section {
            margin-bottom: 25px;
        }

        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }

        .section h3 {
            color: #34495e;
            margin-top: 20px;
        }

        .section h4 {
            color: #555;
        }

        .contact-info {
            background-color: #f8f9fa;
            padding: 15px;
            margin: 20px 0;
        }

        .pricing-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }

        .info-table {
            width: 50%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }

        .info-table th,
        .info-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .pricing-table th,
        .pricing-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .pricing-table th {
            background-color: #f2f2f2;
        }

        .milestone-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }

        .milestone-table th,
        .milestone-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        .milestone-table th {
            background-color: #e8f4fd;
        }

        .checklist {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
        }

        .compliance-box {
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }

        .signature-section {
            border: 2px solid #333;
            padding: 20px;
            margin: 20px 0;
            background-color: #fafafa;
        }
    </style>
</head>

<body>

    <div class="header">
        <h2>VENDOR PROPOSAL</h2>
    </div>

    <div class="contact-info">
    <div>
        <strong>Submitted to:</strong><br>
        {{basic["Client Information"]["company name"]}}<br>
        {{basic['Client Information']['Address'].replace(',','<br>')}}<br>
        Attention: {{basic["Client Information"]["Name"]}}, Procurement Officer<br><br>
    </div>
        <div><strong>Project:</strong> {{basic["Client Information"]["Company"]}}</div><br>
    <div>
        <strong>Submitted by:</strong><br>
        <span class="bracketed">{{basic["Dealer Information"]["company"]}}</span><br>
        <span class="bracketed">{{basic["Dealer Information"]["address"].replace(',','<br>')}}</span><br>
        <span class="bracketed">{{basic["Dealer Information"]["phone"]}}</span> | <span class="bracketed">{{basic["Dealer Information"]["email"]}}</span><br>
        Project Manager: <span class="bracketed">{{basic["Dealer Information"]["name"]}}, {{basic["Dealer Information"]["email"]}}</span><br><br>
    </div>
        <div><strong>Date of Submission:</strong> <span class="bracketed">{{today}}</span></div>
        <div><strong>Proposal Valid Until:</strong> <span class="bracketed">{{basic["Quotation Details"]["Due Date"]}}</span></div><br>
    <div>
        <strong>Authorized Signature:</strong><br>
        <span class="bracketed">{{basic["Dealer Information"]["name"]}}</span><br>
        <span class="bracketed">{{today}}</span>
    </div>
    </div>

    <div class="section">
        <h2 class="section-title">DEALER INFORMATION</h2>

        <div><div class="subsection-title">Primary Dealer Details</div>
        <table class="info-table">
            <tr>
                <th>Field</th>
                <th>Information</th>
            </tr>
            <tr>
                <td>Name</td>
                <td>{{basic["Dealer Information"]["name"]}}</td>
            </tr>
            <tr>
                <td>Company</td>
                <td>{{basic["Dealer Information"]["company"]}}</td>
            </tr>
            <tr>
                <td>Address</td>
                <td>{{' '.join(basic["Dealer Information"]["address"].split(',')[:-1])}}</td>
            </tr>
            <tr>
                <td>City, State, ZIP Code</td>
                <td>{{basic["Dealer Information"]["address"].split(',')[-1]}}</td>
            </tr>
            <tr>
                <td>Phone</td>
                <td>{{basic["Dealer Information"]["phone"]}}</td>
            </tr>
            <tr>
                <td>Email</td>
                <td>{{basic["Dealer Information"]["email"]}}</td>
            </tr>
            <tr>
                <td>Post/Position</td>
                <td>{{basic["Dealer Information"]["post"]}}</td>
            </tr>
        </table>
        </div>

        <div><div class="subsection-title">Charges Covered by Dealer</div>
        <table class="info-table">
            <tr>
                <th>Service Type</th>
                <th>Coverage</th>
                <th>Amount</th>
            </tr>
            <tr>
                <td>Shipment Charges</td>
                <td>Both locations (Dan Kinney & Chesterfield)</td>
                <td><span class="price-placeholder">$ 0</span></td>
            </tr>
            <tr>
                <td>Delivery & Handling</td>
                <td>White glove delivery service</td>
                <td><span class="price-placeholder">$ 0</span> </td>
            </tr>
            <tr>
                <td>Installation Services</td>
                <td>Complete assembly and placement</td>
                <td><span class="price-placeholder">$ 0</span> </td>
            </tr>
            <tr>
                <td>Packaging Removal</td>
                <td>Debris removal and disposal</td>
                <td><span class="price-placeholder">$ 0</span></td>
            </tr>
            <tr>
                <td>Storage (if needed)</td>
                <td>Temporary warehousing</td>
                <td><span class="price-placeholder">$ 0</span></td>
            </tr>
            <tr>
                <td>Expedited Shipping</td>
                <td>Rush delivery if required</td>
                <td><span class="price-placeholder">$ 0</span></td>
            </tr>
        </table>
        </div>

        <div><div class="subsection-title">Additional Services Provided by Dealer</div>
        <ul>
            <li>Pre-delivery inspection and quality control</li>
            <li>Site survey and space planning assistance</li>
            <li>Project coordination and timeline management</li>
            <li>Post-installation support and warranty service</li>
            <li>Furniture reconfiguration services</li>
            <li>Maintenance and repair services</li>
        </ul>
        </div>
        <div><div class="subsection-title">Dealer Authorization & Certifications</div>
        <table class="info-table">
            <tr>
                <th>Category</th>
                <th>Details</th>
            </tr>
            <tr>
                <td>Geographic Service Area</td>
                <td>[Specify coverage area]</td>
            </tr>
            <tr>
                <td>Service Response Time</td>
                <td>[Specify response time for service calls]</td>
            </tr>
            <tr>
                <td>Local Inventory</td>
                <td><input type="checkbox" class="checkbox"> Yes <input type="checkbox" class="checkbox"> No - If yes,
                    specify items maintained in stock: [List items]</td>
            </tr>
        </table>
        </div>

        <div><div class="subsection-title">Manufacturer Authorizations</div>
        <table class="info-table">
            <tr>
                <th>Manufacturer</th>
                <th>Authorization Level</th>
            </tr>
            <tr>
                <td>Manufacturer 1</td>
                <td>[Premium/Standard/Basic]</td>
            </tr>
            <tr>
                <td>Manufacturer 2</td>
                <td>[Premium/Standard/Basic]</td>
            </tr>
            <tr>
                <td>Manufacturer 3</td>
                <td>[Premium/Standard/Basic]</td>
            </tr>
        </table>
        </div>

        <div><div class="subsection-title">Dealer Certifications</div>
        <ul>
            <li>Certified Installation Team</li>
            <li>BIFMA Certified</li>
            <li>OSHA Safety Certified</li>
            <li>Other relevant certifications: [List]</li>
        </ul>
        </div>

        <div><div class="subsection-title">Contact Information for Project Coordination</div>
        <table class="info-table">
            <tr>
                <th>Primary Contact</th>
            </tr>
            <tr>
                <td>
                    <strong>Name:</strong> {{basic["Dealer Information"]["name"]}}<br>
                    <strong>Direct Phone:</strong> {{basic["Dealer Information"]["phone"]}}<br>
                    <strong>Email:</strong> {{basic["Dealer Information"]["email"]}}<br>
                    <strong>Best Time to Reach:</strong> 9:00 AM - 5:00 PM
                </td>
                
            </tr>
        </table>
        </div>
    </div>

    <div class="section">
        <h2>TABLE OF CONTENTS</h2>
        <ul>
            <li>Executive Summary</li>
            <li>Company Qualifications</li>
            <li>Technical Proposal</li>
            <li>Detailed Pricing Schedule</li>
            <li>Project Management & Timeline</li>
            <li>Installation Plan</li>
            <li>Warranty & Service</li>
            <li>Quality Assurance</li>
            <li>Sustainability Commitment</li>
            <li>References & Experience</li>
            <li>Insurance & Compliance</li>
            <li>Appendices</li>
        </ul>
    </div>

    <div class="section">
        <h2>1. EXECUTIVE SUMMARY</h2>

        <div><h3>Project Understanding</h3>
        <p>We understand that IOM Washington DC requires furniture procurement for two family center locations with a
            firm delivery deadline of {{basic['Quotation Details']['Due Date']}}. Our proposal addresses {{basic["furniture_items_and_pricing"]|length}} furniture items specified across
            the {{basic['Client Information']['Company']}}.</p>
        </div>
        <div><h3>Company Overview</h3>
        {% for code,ent_data in progress.items() %}
        <h4> {{ent_data["Enterprise Information"]["name"]}}</h4>

        <p>{{ent_data["Enterprise Information"]["description"]}}</p>
        {% endfor %}
        </div>
        <div><h3>Key Commitments</h3>
        <ul>
            <li><strong>On-Time Delivery:</strong> Guaranteed completion by {{basic['Quotation Details']['Due Date']}}</li>
            <li><strong>Quality Assurance:</strong> All Grade A contract furniture meeting specifications</li>
            <li><strong>Full Service:</strong> Complete procurement, delivery, and installation</li>
            <li><strong>Warranty:</strong> 10+ year comprehensive warranty on all items</li>
            <li><strong>Local Support:</strong> <span class="bracketed">[Location]</span> service team for ongoing
                support</li>
        </ul>
        </div>

        <div><h3>Total Project Investment</h3>
        <table class="pricing-table">
            <tr>
                <td>Furniture Subtotal:</td>
                <td><span class="bracketed">$ {{basic["sub_total"]}}</span></td>
            </tr>
            <tr>
                <td>Delivery & Installation:</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Project Management:</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Additional Services:</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr style="font-weight: bold;">
                <td>TOTAL PROJECT COST</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
        </table>
        </div>
        <p><em>Tax-exempt for IOM</em></p>

        <div><h3>Primary Partners</h3>
        <ul>
            <li>Project Manager: <span class="bracketed">[Name, Credentials]</span></li>
            <li>Installation Supervisor: <span class="bracketed">[Name]</span></li>
            <li>Quality Control Manager: <span class="bracketed">[Name]</span></li>
        </ul>
        </div>
    </div>

    <div class="section">
        <h2>2. COMPANY QUALIFICATIONS</h2>

        <div><h3>Company Information</h3>
        <ul>
            <li><strong>Company Name:</strong> <span class="bracketed">{{basic["Dealer Information"]["company"]}}</span></li>
            <li><strong>Years in Business:</strong> <span class="bracketed">[X]</span> years</li>
            <li><strong>Business Registration:</strong> State Corporation #<span class="bracketed">[Number]</span></li>
            <li><strong>Federal Tax ID:</strong> <span class="bracketed">[EIN Number]</span></li>
            <li><strong>DUNS Number:</strong> <span class="bracketed">[Number]</span></li>
            <li><strong>Annual Revenue:</strong> <span class="bracketed">$ 0</span> (3-year average)</li>
        </ul>
        </div>

        <div><h3>Relevant Experience</h3>
        <ul>
            <li>Total Contract Furniture Projects: <span class="bracketed">[X]</span> projects</li>
            <li>Combined Project Value: <span class="bracketed">$[X]</span> million</li>
            <li>Average Project Completion Time: <span class="bracketed">[X]</span>% on-time delivery rate</li>
            <li>Client Satisfaction Rate: <span class="bracketed">[X]</span>%</li>
        </ul>
        </div>

        <div><h3>Authorized Manufacturer Relationships</h3>
        <ul>
            <li>Manufacturer 1 - Authorized Dealer since <span class="bracketed">[Year]</span></li>
            <li>Manufacturer 2 - Authorized Dealer since <span class="bracketed">[Year]</span></li>
            <li>Manufacturer 3 - Authorized Dealer since <span class="bracketed">[Year]</span></li>
        </ul>
        </div>

        <div><h3>Key Personnel</h3>
        <h4>Project Manager: <span class="bracketed">[Name, Credentials]</span></h4>
        <ul>
            <li><span class="bracketed">[X]</span> years contract furniture experience</li>
            <li><span class="bracketed">[Relevant certifications]</span></li>
            <li>Contact: <span class="bracketed">[Phone/Email]</span></li>
        </ul>

        <h4>Installation Supervisor: <span class="bracketed">[Name]</span></h4>
        <ul>
            <li><span class="bracketed">[X]</span> years installation experience</li>
            <li><span class="bracketed">[Safety certifications]</span></li>
        </ul>

        <h4>Quality Control Manager: <span class="bracketed">[Name]</span></h4>
        <ul>
            <li><span class="bracketed">[X]</span> years QA experience</li>
            <li><span class="bracketed">[Relevant qualifications]</span></li>
        </ul>
        </div>

        <div><h3>Financial Capacity</h3>
        <ul>
            <li><strong>Bonding Capacity:</strong> <span class="bracketed">$ 0</span></li>
            <li><strong>Credit Rating:</strong> <span class="bracketed">[Rating]</span></li>
            <li><strong>Bank References:</strong> <span class="bracketed">[Bank Name, Contact]</span></li>
            <li><strong>Insurance Coverage:</strong> Details per Section 11</li>
        </ul>
        </div>
    </div>

    <div class="section">
        <h2>3. TECHNICAL PROPOSAL</h2>

        <h3>3.1 Product Compliance Summary</h3>
        <p>All proposed furniture meets or exceeds RFQ specifications and includes:</p>
        <ul>
            <li>Contract-grade construction for institutional use</li>
            <li>Compliance with all dimensional requirements</li>
            <li>Specified color and finish requirements</li>
            <li>Grade A quality standards</li>
        </ul>
        {% for code,ent_data in progress.items() %}
        <h3><u>{{ent_data["Enterprise Information"]["name"]}}</u>:</h3>
        <ul>
        {% for items in ent_data["furniture_items_and_pricing"]%}
        <li>
        <h4>{{items["reference"]}} ({{items["quantity"]}} units)</h4>
        <p><strong>Proposed Product:</strong> <span class="bracketed">{{items["product code"]}} - {{items["description"]}}</span></p>
        {% if items["attributes"] %}
        <p><strong>Specifications:</strong></p>
        
        <ul>
        {% for att in items["attributes"] %}
            <li> {{att}} </li>
        {% endfor %}
        </ul>
        {% endif %}
        {% endfor %}
        </li>
        </ul>
    
    {% endfor %}
    </div>
    <div class="section">
        <h2>4. DETAILED PRICING SCHEDULE</h2>

        <h3>4.1 Furniture Pricing Summary</h3>
        {% for code,ent_data in progress.items() %}
        <h3><u>{{ent_data["Enterprise Information"]["name"]}}</u></h3>
        <table class="pricing-table">
            <tr>
                <th>Description</th>
                <th>Quantity</th>
                <th>Extended Price</th>
            </tr>
            {% for item in ent_data["furniture_items_and_pricing"] or [] %}
            <tr>
                <td>{{ item["RFP_description"] }}</td>
                <td>{{ item["quantity"] }}</td>
                <td>${{ item["total amount"] }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}

        <div><h3>4.2 Additional Services</h3>
        <table class="pricing-table">
            <tr>
                <th>Service Description</th>
                <th>Price</th>
            </tr>
            <tr>
                <td>Delivery (F.O.B. Destination, both locations)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Installation (Complete assembly and placement)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Project Management (Coordination and oversight)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Debris Removal (Packaging disposal)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr style="font-weight: bold;">
                <td>SERVICES SUBTOTAL</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
        </table>
        </div>

        <div><h3>4.3 Optional Services (As Requested)</h3>
        <table class="pricing-table">
            <tr>
                <th>Service Description</th>
                <th>Estimated Price</th>
            </tr>
            <tr>
                <td>Data Network Cable Management (Pass-through requirements)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Electrical Cable Management (Cord management solutions)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Planters (Per test fit requirements)</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
        </table>
        </div>

        <div><h3>4.4 Project Total Summary</h3>
        <table class="pricing-table">
            <tr>
                <th>Component</th>
                <th>Amount</th>
            </tr>
            <tr>
                <td>Furniture Subtotal</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Services Subtotal</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr>
                <td>Optional Services</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
            <tr style="font-weight: bold; font-size: 1.2em;">
                <td>TOTAL PROJECT COST</td>
                <td><span class="bracketed">$ 0</span></td>
            </tr>
        </table>
        </div>
        <p><strong>Note:</strong> IOM is tax-exempt - no sales tax applied</p>

        <div><h3>4.5 Payment Terms</h3>
        <ul>
            <li><strong>Terms:</strong> Net 30 days from delivery and acceptance</li>
            <li><strong>Progress Payments:</strong> <span class="bracketed">[If applicable]</span></li>
            <li><strong>Final Payment:</strong> Upon completion and IOM acceptance</li>
            <li><strong>Currency:</strong> USD</li>
            <li><strong>Early Payment Discount:</strong> 2% if paid within 10 days</li>
        </ul>
        </div>
    </div>

    <div class="section">
        <h2>5. PROJECT MANAGEMENT & TIMELINE</h2>

        <div><h3>5.1 Project Timeline</h3>
        <p><strong>Phase 1:</strong> Order Processing & Manufacturing</p>
        <p><strong>Phase 2:</strong> Delivery & Installation</p>

        <ul>
            <li><strong>Week 1:</strong> Contract execution and order placement</li>
            <li><strong>Week 2:</strong> Shop drawings and final approvals</li>
            <li><strong>Weeks 3-8:</strong> Manufacturing and quality control</li>
            <li><strong>Week 8:</strong> Pre-delivery inspection</li>
            <li><strong>Week 9:</strong> Delivery coordination and staging</li>
            <li><strong>Week 10:</strong> Installation and completion</li>
        </ul>

        <p><strong>Final Completion:</strong> September 30, 2025 (1 day ahead of deadline)</p>
        </div>

        <div><h3>5.2 Critical Milestones</h3>
        <table class="milestone-table">
            <tr>
                <th>Milestone</th>
                <th>Target Date</th>
            </tr>
            {% for timeline in basic["project_timeline"] %}
            <tr>
            <td>{{timeline["milestone"]}}</td>
            <td>{{timeline["date"]}}</td>
            </tr>
            {% endfor %}
        </table>
        </div>

        <div><h3>5.3 Risk Management</h3>
        <p><strong>Identified Risks and Mitigation:</strong></p>
        <ul>
            <li><strong>Manufacturing Delays:</strong> Built-in 1-week buffer in schedule</li>
            <li><strong>Shipping Issues:</strong> Multiple shipping options and expedited delivery available</li>
            <li><strong>Installation Challenges:</strong> Pre-site survey and coordination meetings</li>
            <li><strong>Quality Issues:</strong> Comprehensive QC process at factory and delivery</li>
        </ul>
        </div>

        <div><h3>5.4 Project Communication</h3>
        <ul>
            <li><strong>Weekly Progress Reports:</strong> Every Friday via email</li>
            <li><strong>Milestone Updates:</strong> Real-time notifications for key events</li>
            <li><strong>Issue Escalation:</strong> 24-hour response for critical issues</li>
            <li><strong>Primary Contact:</strong> <span class="bracketed">[Project Manager Name, Phone, Email]</span>
            </li>
        </ul>
        </div>
    </div>

    <div class="section">
        <h2>6. INSTALLATION PLAN</h2>

        <div><h3>6.1 Pre-Installation Activities</h3>
        <div><p><strong>Site Survey:</strong></p>
        <ul>
            <li>Verify dimensions and access routes</li>
            <li>Identify potential installation challenges</li>
            <li>Coordinate with facility management</li>
            <li>Document existing conditions</li>
        </ul></div>

        <div><p><strong>Logistics Coordination:</strong></p>
        <ul>
            <li>Schedule delivery appointments (2-day advance notice)</li>
            <li>Arrange appropriate delivery vehicles</li>
            <li>Confirm installation crew and tools</li>
            <li>Obtain necessary facility permits</li>
        </ul></div></div>

        <div><h3>6.2 Installation Process</h3>
        <p><strong>Day 1: Dan Kinney Family Center</strong></p>
        <ul>
            <li><strong>Morning:</strong> Delivery and staging of all items</li>
            <li><strong>Afternoon:</strong> Installation of community room furniture</li>
            <li><strong>Evening:</strong> Begin childcare area installation</li>
        </ul>

        <p><strong>Day 2: Complete Installation</strong></p>
        <ul>
            <li><strong>Morning:</strong> Finish Dan Kinney childcare installation</li>
            <li><strong>Afternoon:</strong> Chesterfield delivery and installation</li>
            <li><strong>Evening:</strong> Final quality inspection and cleanup</li>
        </ul>
        </div>

        <div><h3>6.3 Installation Team</h3>
        <div><p><strong>Team Composition:</strong></p>
        <ul>
            <li><strong>Installation Supervisor:</strong> <span class="bracketed">[Name, Credentials]</span></li>
            <li><strong>Lead Installers:</strong> <span class="bracketed">[Number]</span> certified technicians</li>
            <li><strong>Quality Inspector:</strong> <span class="bracketed">[Name, Experience]</span></li>
            <li><strong>Project Coordinator:</strong> On-site for duration</li>
        </ul></div>

        <div><p><strong>Safety Protocols:</strong></p>
        <ul>
            <li>All installers OSHA 30-hour certified</li>
            <li>Comprehensive PPE required</li>
            <li>Site safety meeting before starting</li>
            <li>Daily safety briefings</li>
        </ul></div></div>
        <div><h3>6.4 Quality Control During Installation</h3>
        <ul>
            <li>Pre-installation inspection of all items</li>
            <li>Assembly verification at each step</li>
            <li>Final placement and alignment check</li>
            <li>Punch list creation and resolution</li>
            <li>Client walkthrough and approval</li>
        </ul></div>
    </div>
    <div class="section">
        <h2>7. WARRANTY & SERVICE</h2>
        <div><h3>7.1 Comprehensive Warranty Coverage</h3>
        <div><p><strong>Furniture Warranty Terms:</strong></p>
        <ul>
            <li><strong>Duration:</strong> 10 years minimum (exceeds RFQ requirement)</li>
            <li><strong>Coverage:</strong> Parts, labor, and travel included</li>
            <li><strong>Usage Rating:</strong> 40+ hours per week commercial use</li>
            <li><strong>Response Time:</strong> 48-hour response (exceeds 48-72 requirement)</li>
        </ul></div>

        <div><p><strong>Specific Warranty by Item Type:</strong></p>
        <ul>
            <li><strong>Chairs:</strong> 10 years comprehensive, 5 years fabric</li>
            <li><strong>Tables:</strong> 10 years structure, 5 years surface</li>
            <li><strong>Storage:</strong> 10 years comprehensive including hardware</li>
            <li><strong>Rockers:</strong> 10 years frame, 3 years fabric/cushions</li>
        </ul></div></div>

        <div><h3>7.2 Local Service Support</h3>
        <div><p><strong>Service Team:</strong></p>
        <ul>
            <li><strong>Service Manager:</strong> <span class="bracketed">[Name, Contact]</span></li>
            <li><strong>Field Technicians:</strong> <span class="bracketed">[Number]</span> certified locally</li>
            <li><strong>Service Territory:</strong> 100-mile radius of project</li>
            <li><strong>Parts Inventory:</strong> Local stock maintained</li>
        </ul></div>

        <div><p><strong>Service Commitments:</strong></p>
        <ul>
            <li><strong>Emergency Response:</strong> 24 hours</li>
            <li><strong>Standard Service:</strong> 48 hours (meets requirement)</li>
            <li><strong>Routine Maintenance:</strong> Scheduled within 1 week</li>
            <li><strong>Warranty Claims:</strong> Same-day processing</li>
        </ul></div></div>

        <div><h3>7.3 Warranty Documentation</h3>
        <p><strong>Provided at Project Completion:</strong></p>
        <ul>
            <li>Individual item warranty certificates</li>
            <li>Manufacturer authorization documentation</li>
            <li>Care and maintenance instructions</li>
            <li>Warranty registration confirmation</li>
            <li>Local service contact information</li>
        </ul></div>

        <div><h3>7.4 Post-Installation Support</h3>
        <ul>
            <li><strong>30-Day Follow-up:</strong> Comprehensive project review</li>
            <li><strong>90-Day Check:</strong> Usage assessment and adjustment</li>
            <li><strong>Annual Inspections:</strong> Preventive maintenance available</li>
            <li><strong>Training:</strong> End-user furniture care and adjustment</li>
        </ul></div>
    </div>
    <div class="section">
        <h2>8. QUALITY ASSURANCE</h2>
        <div><h3>8.1 Quality Control Process</h3>
        <div><p><strong>Factory Quality Control:</strong></p>
        <ul>
            <li>Incoming material inspection</li>
            <li>In-process manufacturing checks</li>
            <li>Final assembly verification</li>
        </ul></div>

        <div><p><strong>Pre-Delivery Inspection:</strong></p>
        <ul>
            <li>Pre-shipment quality review</li>
            <li>Complete item-by-item inspection</li>
            <li>Packaging and protection verification</li>
            <li>Documentation and certification</li>
            <li>Damage-free delivery guarantee</li>
        </ul></div>

        <div><p><strong>Installation Quality Control:</strong></p>
        <ul>
            <li>Delivery condition assessment</li>
            <li>Proper assembly verification</li>
            <li>Placement and alignment checks</li>
            <li>Final quality walkthrough</li>
        </ul></div></div>

        <div><h3>8.2 Quality Standards</h3>
        <div><p><strong>Manufacturing Standards:</strong></p>
        <ul>
            <li>BIFMA compliance for all items</li>
            <li>Contract-grade construction requirements</li>
            <li>Specified dimensional tolerances</li>
            <li>Finish quality standards</li>
        </ul></div>

        <div><p><strong>Installation Standards:</strong></p>
        <ul>
            <li>Level, plumb, and square installation</li>
            <li>Proper component alignment</li>
            <li>Secure fastening and stability</li>
            <li>Clean and complete presentation</li>
        </ul></div></div>

        <div><h3>8.3 Quality Assurance Team</h3>
        <ul>
            <li><strong>QA Manager:</strong> <span class="bracketed">[Name, Qualifications]</span></li>
            <li><strong>Factory Inspector:</strong> <span class="bracketed">[Name, Experience]</span></li>
            <li><strong>Installation Inspector:</strong> <span class="bracketed">[Name, Credentials]</span></li>
            <li><strong>Client Liaison:</strong> <span class="bracketed">[Name, Contact Information]</span></li>
        </ul></div>

        <div><h3>8.4 Continuous Improvement</h3>
        <ul>
            <li>Client feedback integration</li>
            <li>Process improvement documentation</li>
            <li>Best practices sharing</li>
            <li>Quality metrics tracking</li>
        </ul></div>
    </div>
    <div class="section">
        <h2>9. SUSTAINABILITY COMMITMENT</h2>
        <div><h3>9.1 Environmental Responsibility</h3>
        <div><p><strong>Manufacturing Partners:</strong> All manufacturers demonstrate environmental stewardship through:</p>
        <ul>
            <li>ISO 14001 environmental management certification</li>
            <li>Sustainable material sourcing programs</li>
            <li>Waste reduction and recycling initiatives</li>
            <li>Energy-efficient manufacturing processes</li>
        </ul></div>

        <div><p><strong>Product Sustainability:</strong></p>
        <ul>
            <li>GREENGUARD Gold certification available</li>
            <li>Low-emission materials and finishes</li>
            <li>Recyclable content in products</li>
            <li>End-of-life recycling programs</li>
        </ul></div></div>

        <div><h3>9.2 Sustainable Practices</h3>
        <div><p><strong>Packaging and Delivery:</strong></p>
        <ul>
            <li>Minimal packaging materials</li>
            <li>Recyclable packaging components</li>
            <li>Efficient delivery routing</li>
            <li>Packaging material removal and recycling</li>
        </ul></div>

        <div><p><strong>Installation Process:</strong></p>
        <ul>
            <li>Waste minimization during installation</li>
            <li>Proper disposal of installation materials</li>
            <li>Energy-efficient installation practices</li>
            <li>Local sourcing when possible</li>
        </ul></div></div>

        <div><h3>9.3 Long-Term Sustainability</h3>
        <div><p><strong>Product Longevity:</strong></p>
        <ul>
            <li>Durable construction for extended service life</li>
            <li>Repairable and maintainable designs</li>
            <li>Timeless styling to avoid premature replacement</li>
        </ul></div>

        <div><p><strong>Service Sustainability:</strong></p>
        <ul>
            <li>Local service reduces travel impact</li>
            <li>Preventive maintenance extends product life</li>
            <li>Repair-first approach reduces waste</li>
        </ul></div></div>
    </div>
    <div class="section">
        <h2>10. REFERENCES & EXPERIENCE</h2>
        <div><h3>10.1 Similar Project Experience</h3>

        <div><h4>Project 1: Educational Institution Furniture</h4>
        <ul>
            <li><strong>Client:</strong> <span class="bracketed">[Institution Name]</span></li>
            <li><strong>Value:</strong> <span class="bracketed">$ 0</span></li>
            <li><strong>Completion:</strong> <span class="bracketed">[Date]</span></li>
            <li><strong>Scope:</strong> <span class="bracketed">[X]</span> items of contract furniture for multiple
                locations</li>
            <li><strong>Contact:</strong> <span class="bracketed">[Name, Title, Phone, Email]</span></li>
            <li><strong>Results:</strong> Completed 1 week early, 100% client satisfaction</li>
        </ul></div>

        <div><h4>Project 2: Government Facility Furniture</h4>
        <ul>
            <li><strong>Client:</strong> <span class="bracketed">[Agency Name]</span></li>
            <li><strong>Value:</strong> <span class="bracketed">$ 0</span></li>
            <li><strong>Completion:</strong> <span class="bracketed">[Date]</span></li>
            <li><strong>Scope:</strong> <span class="bracketed">[X]</span> items including childcare and office
                furniture</li>
            <li><strong>Contact:</strong> <span class="bracketed">[Name, Title, Phone, Email]</span></li>
            <li><strong>Results:</strong> Zero punch list items, exceeded quality expectations</li>
        </ul></div>

        <div><h4>Project 3: Healthcare Facility Furniture</h4>
        <ul>
            <li><strong>Client:</strong> <span class="bracketed">[Facility Name]</span></li>
            <li><strong>Value:</strong> <span class="bracketed">$ 0</span></li>
            <li><strong>Completion:</strong> <span class="bracketed">[Date]</span></li>
            <li><strong>Scope:</strong> <span class="bracketed">[X]</span> items with strict timeline requirements</li>
            <li><strong>Contact:</strong> <span class="bracketed">[Name, Title, Phone, Email]</span></li>
            <li><strong>Results:</strong> Met aggressive deadline, comprehensive warranty service</li>
        </ul></div></div>

        <div><h3>10.2 Client References</h3>

        <div class="contact-info">
            <h4>Reference 1</h4>
            <p><span class="bracketed">[Organization Name]</span><br>
                <span class="bracketed">[Contact Person, Title]</span><br>
                <span class="bracketed">[Address]</span><br>
                <span class="bracketed">[Phone]</span> | <span class="bracketed">[Email]</span><br>
                <strong>Project Value:</strong> <span class="bracketed">$ 0</span> | <strong>Completion:</strong>
                <span class="bracketed">[Date]</span>
            </p>
        </div>

        <div class="contact-info">
            <h4>Reference 2</h4>
            <p><span class="bracketed">[Organization Name]</span><br>
                <span class="bracketed">[Contact Person, Title]</span><br>
                <span class="bracketed">[Address]</span><br>
                <span class="bracketed">[Phone]</span> | <span class="bracketed">[Email]</span><br>
                <strong>Project Value:</strong> <span class="bracketed">$ 0</span> | <strong>Completion:</strong>
                <span class="bracketed">[Date]</span>
            </p>
        </div>

        <div class="contact-info">
            <h4>Reference 3</h4>
            <p><span class="bracketed">[Organization Name]</span><br>
                <span class="bracketed">[Contact Person, Title]</span><br>
                <span class="bracketed">[Address]</span><br>
                <span class="bracketed">[Phone]</span> | <span class="bracketed">[Email]</span><br>
                <strong>Project Value:</strong> <span class="bracketed">$ 0</span> | <strong>Completion:</strong>
                <span class="bracketed">[Date]</span>
            </p>
        </div></div>

        <div><h3>10.3 Performance Metrics</h3>
        <ul>
            <li><strong>On-Time Delivery Rate:</strong> <span class="bracketed">[X]</span>% (last 3 years)</li>
            <li><strong>Quality Satisfaction Score:</strong> <span class="bracketed">[X]</span>/10 average</li>
            <li><strong>Repeat Client Rate:</strong> <span class="bracketed">[X]</span>%</li>
            <li><strong>Warranty Claim Rate:</strong> Less than <span class="bracketed">[X]</span>%</li>
            <li><strong>Safety Record:</strong> Zero incidents (last 5 years)</li>
        </ul></div>
    </div>
    <div class="section">
        <h2>11. INSURANCE & COMPLIANCE</h2>
        <h3>11.1 Insurance Coverage</h3>

        <div><h4>Current Insurance Policies:</h4>

        <div><h5>General Liability:</h5>
        <ul>
            <li><strong>Coverage:</strong> $2,000,000 per occurrence</li>
            <li><strong>Aggregate:</strong> $4,000,000 annual</li>
            <li><strong>Carrier:</strong> <span class="bracketed">[Insurance Company]</span></li>
            <li><strong>Policy #:</strong> <span class="bracketed">[Number]</span></li>
            <li><strong>Expiration:</strong> <span class="bracketed">[Date]</span></li>
        </ul></div>

        <div><h5>Workers' Compensation:</h5>
        <ul>
            <li><strong>Coverage:</strong> As required by state law</li>
            <li><strong>Employer Liability:</strong> $1,000,000</li>
            <li><strong>Carrier:</strong> <span class="bracketed">[Insurance Company]</span></li>
            <li><strong>Policy #:</strong> <span class="bracketed">[Number]</span></li>
        </ul></div>

        <div><h5>Commercial Auto:</h5>
        <ul>
            <li><strong>Coverage:</strong> $1,000,000 combined single limit</li>
            <li><strong>Carrier:</strong> <span class="bracketed">[Insurance Company]</span></li>
            <li><strong>Policy #:</strong> <span class="bracketed">[Number]</span></li>
        </ul></div>

        <div><h5>Property Coverage:</h5>
        <ul>
            <li><strong>Coverage:</strong> Full replacement value during project</li>
            <li><strong>Transit Coverage:</strong> Door-to-door protection</li>
            <li><strong>Installation Coverage:</strong> On-site protection</li>
        </ul></div></div>

        <div><h3>11.2 Licensing & Certifications</h3>

        <div><h4>Business Licenses:</h4>
        <ul>
            <li><strong>State Business License:</strong> #<span class="bracketed">[Number]</span>, Expires <span
                    class="bracketed">[Date]</span></li>
            <li><strong>Federal Tax ID:</strong> <span class="bracketed">[EIN]</span></li>
            <li><strong>Professional Certifications:</strong> <span class="bracketed">[List relevant
                    certifications]</span></li>
        </ul></div>

        <div><h4>Safety Certifications:</h4>
        <ul>
            <li>OSHA 30-Hour Construction Safety</li>
            <li>Manufacturer Installation Certifications</li>
            <li>Safety Training Documentation</li>
        </ul></div></div>

        <div><h3>11.3 Compliance Commitments</h3>

        <div><h4>Facility Compliance:</h4>
        <ul>
            <li>Full adherence to IOM facility security procedures</li>
            <li>Background checks for personnel if required</li>
            <li>Site safety protocol compliance</li>
            <li>Environmental protection measures</li>
        </ul></div>

        <div><h4>Quality Compliance:</h4>
        <ul>
            <li>All products meet specified standards</li>
            <li>Installation per manufacturer requirements</li>
            <li>Code compliance verification</li>
            <li>Final inspection and approval</li>
        </ul></div></div>
    </div>
    <div class="section">
        <div><h2>12. APPENDICES</h2>
        <div><h3>Appendix A: Required Documentation</h3>
        <ul>
            <li>Business Registration Certificate</li>
            <li>Insurance Certificates (all policies)</li>
            <li>Financial Statements (last 3 years)</li>
            <li>Manufacturer Authorization Letters</li>
            <li>Professional Licenses and Certifications</li>
        </ul></div>

        <div><h3>Appendix B: Technical Specifications</h3>
        <ul>
            <li>Complete Product Specification Sheets (all 87 items)</li>
            <li>Manufacturer Technical Data</li>
            <li>Installation Instructions</li>
            <li>Care and Maintenance Guidelines</li>
            <li>Warranty Terms and Conditions</li>
        </ul></div>

        <div><h3>Appendix C: Project Documentation</h3>
        <ul>
            <li>Sample Timeline and Milestone Charts</li>
            <li>Quality Control Checklists</li>
            <li>Safety Procedures and Protocols</li>
            <li>Installation Process Documentation</li>
        </ul></div>

        <div><h3>Appendix D: Company Information</h3>
        <ul>
            <li>Organizational Chart</li>
            <li>Key Personnel Resumes</li>
            <li>Company Brochure and Capabilities</li>
            <li>Awards and Recognition</li>
            <li>Client Testimonials</li>
        </ul></div>

        <div><h3>Appendix E: Visual Materials</h3>
        <ul>
            <li>Product Photography (proposed items)</li>
            <li>Installation Process Photos (from similar projects)</li>
            <li>Completed Project Photos</li>
            <li>3D Renderings (if applicable)</li>
        </ul></div></div>
    </div>
    <div class="section">
        <h2>PROPOSAL SUBMISSION CHECKLIST</h2>
        <div class="checklist">
            <div><h3>Required for Submission:</h3>
            <ul>
                <li>Completed Cover Page with authorized signature</li>
                <li>Executive Summary addressing all key requirements</li>
                <li>Company qualifications and experience documentation</li>
                <li>Technical proposal with complete product specifications</li>
                <li>Detailed pricing schedule with all costs included</li>
                <li>Project timeline demonstrating {{basic['Quotation Details']['Due Date']}} completion</li>
                <li>Installation plan and methodology</li>
                <li>Warranty documentation (10+ year terms)</li>
                <li>Quality assurance procedures</li>
                <li>Sustainability commitments and practices</li>
                <li>Client references with contact information</li>
                <li>Insurance certificates and compliance documentation</li>
                <li>All required appendices and supporting materials</li>
            </ul></div>

            <div><h3>Submission Details:</h3>
            <ul>
                <li><strong>Format:</strong> PDF, maximum 10MB per file</li>
                <li><strong>Email:</strong> procurement@iom.int</li>
                <li><strong>Subject:</strong> "RFQ Response - {{basic['Client Information']['RFP Number']}} </li>
                <li><strong>Deadline:</strong> {{basic["Quotation Details"]["Due Date"]}}</li>
                <li><strong>Confirmation:</strong> Request delivery receipt confirmation</li>
            </ul></div>
        </div>
    </div>
    <div class="compliance-box">
        <h2>DECLARATION OF COMPLIANCE</h2>
        <div><p>By submitting this proposal, <span class="bracketed">{{basic["Dealer Information"]["company"]}}</span> certifies that:</p>
        <ul>
            <li>We have read and understand all RFQ requirements</li>
            <li>We can meet the {{basic['Quotation Details']['Due Date']}} delivery deadline</li>
            <li>All proposed furniture meets or exceeds specifications</li>
            <li>We accept all terms and conditions as stated in the RFQ</li>
            <li>Our pricing is firm for the entire project duration (120 days)</li>
            <li>We have the financial and operational capacity to complete this project</li>
            <li>All information provided is accurate and complete</li>
        </ul></div>
    </div>
    <div class="signature-section">
        <h3>Authorized Signature:</h3>
        <p><strong>Printed Name:</strong> <span class="bracketed">{{basic["Dealer Information"]["name"]}}</span></p>
        <p><strong>Title:</strong> <span class="bracketed">{{basic["Dealer Information"]["post"]}}</span></p>
        <p><strong>Date:</strong> <span class="bracketed">{{today}}</span></p>
    </div>

</body>
    

</html>
    """

    template = Template(html_template)
    return {"template":template.render(progress=progress,today = today,basic=basic),"data":progress}

def render_quotation_for_enterprise(progress: dict,today) -> str:
    total_amount=0
    for item in progress.get("furniture_items_and_pricing"):
        item["total amount"] = item.get("quantity") * item.get("unit price")
        total_amount+=float(item.get("total amount"))
        # split description into parts
        parts = [p.strip() for p in item["description"].replace(",", "|").split("|") if p.strip()]
        item["reference"] = parts[0]
        item["attributes"] = parts[1:]

    date_obj = datetime.strptime(progress["Quotation Details"]["Issue date"], "%B %d, %Y")

    # Get the day name
    day_name = date_obj.strftime("%A").upper()
    progress["Quotation Details"]["Issue day"] = day_name
    progress["totals"] = {
    "subtotal": total_amount,
    "grand_total": total_amount
}
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RFQ #023-2013 - Furniture for Dan Kinney and Chesterfield Family Centers</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.4;
                color: #333;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .rfq-number {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .section {
                margin-bottom: 25px;
            }
            .section-title {
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 15px;
                color: #2c5aa0;
                border-bottom: 2px solid #2c5aa0;
                padding-bottom: 5px;
            }
            .contact-info {
                background-color: #f5f5f5;
                padding: 15px;
                border-left: 4px solid #2c5aa0;
                margin: 15px 0;
                display: flex;
                justify-content: space-around;
            }
            .requirements-list {
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
            }
            .requirements-list ul {
                margin: 0;
                padding-left: 20px;
            }
            .important-notice {
                background-color: #fffbcc;
                border: 1px solid #f0c040;
                padding: 15px;
                margin: 15px 0;
                border-radius: 5px;
            }
            .quotation-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .quotation-table th,
            .quotation-table td {
                border: 1px solid #333;
                padding: 8px;
                text-align: left;
                vertical-align: top;
            }
            .quotation-table th {
                background-color: #2c5aa0;
                color: white;
                font-weight: bold;
            }
            /* REMOVE automatic header repetition */
            @media print {
                .quotation-table thead {
                    display: table-row-group !important;
                }
                .quotation-table tr {
                    page-break-inside: avoid !important;
                }
            }
            .group-header {
                background-color: #e6f2ff;
                font-weight: bold;
            }
            .specifications-item {
                background-color: #f9f9f9;
                margin: 10px 0;
                padding: 10px;
                border-left: 3px solid #2c5aa0;
            }
            .form-section {
                background-color: #f5f5f5;
                padding: 15px;
                margin: 20px 0;
                border: 2px solid #2c5aa0;
            }
            .checkbox-list {
                list-style-type: none;
                padding-left: 0;
            }
            .checkbox-list li {
                margin: 10px 0;
            }
            .signature-line {
                border-bottom: 1px solid #333;
                display: inline-block;
                min-width: 200px;
                margin: 0 10px;
            }
            .page-break {
                border-top: 2px dashed #ccc;
                margin: 30px 0;
                padding-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{progress["Client Information"]["company name"]}}</h1>
            <h2>REQUEST FOR QUOTATION</h2>
            <div class="rfq-number">RFQ No: {{progress["Quotation Details"]["Quotation ID"]}}</div>
            <p><strong>THIS IS NOT AN ORDER</strong></p>
        </div>

        <div class="contact-info">
            <p><strong>TO:</strong> {{progress["Client Information"]["Name"]}}<br>
            {{progress["Client Information"]["Company"].replace(',','<br>')}}<br>
            {{''.join(progress["Client Information"]["Address"].split(',')[:-2])}}<br>
            {{''.join(progress["Client Information"]["Address"].split(',')[-2:])}}</p>
            
            <p><strong>Date Issued:</strong> {{progress["Quotation Details"]["Issue date"]}}<br>
            <strong>Buyer's Email:</strong> {{progress["Client Information"]["email"]}} <br>
            <strong>Telephone Number:</strong> 417-864-1621<br>
            <strong>DUE DATE:</strong> {{progress["Quotation Details"]["Due Date"]}}</p>
        </div>

        <div class="important-notice">
            <p><strong>QUOTATIONS MUST BE PHYSICALLY RECEIVED IN THE DIVISION OF PURCHASES PRIOR TO {{progress["Quotation Details"]["Due Time"]}} ON {{progress["Quotation Details"]["Issue day"]}}, {{progress["Quotation Details"]["Due Date"]}}.</strong></p>
        </div>

        <div class="requirements-list">
            <ul>
                <li>Quotations shall be submitted on the forms provided and must be manually signed.</li>
                <li>Quotations shall be submitted with the RFQ number clearly indicated.</li>
                <li>Quotations and all required documentation may be faxed to Fax #417-864-1927.</li>
                <li>Quotations received after the bid opening date and time shall be rejected.</li>
                <li>The attached Terms and Conditions shall become part of any purchase order resulting from this RFQ.</li>
            </ul>
        </div>

        <div class="section">
            <div class="section-title">DESCRIPTION</div>
            <h3>FURNITURE FOR {{progress["Client Information"]["Company"]}}</h3>
            <p>See attached General Conditions, Specifications, and Quotation Form for detailed information.</p>
            
            <p><strong>DELIVERY:</strong> F.O.B. DESTINATION<br>
            The articles to be furnished hereunder shall be delivered all transportation charges paid by the bidder to destination.</p>
        </div>

        <div class="page-break">
            <div class="section-title">INSTRUCTION TO BIDDERS</div>
            
            <div class="section">
                <h4>01. Opening Location</h4>
                <p>The Quotations will be opened at the {{progress['Client Information']['Company']}}, {{progress['Client Information']['Address']}} in the presence of Purchasing officials at the due date and time indicated on the RFQ. All bidders or their representatives are invited to attend the opening of the RFQ.</p>
            </div>

            <div class="section">
                <h4>02. RFQ Delivery Requirements</h4>
                <p>Any Quotations received after the above stated time and date will not be considered. It shall be the sole responsibility of the bidder to have their Quotation delivered to the Division of Purchases for receipt on or before the due date and time indicated. If a Quotation is sent by U.S. Mail, the bidder shall be responsible for its timely delivery to the Division of Purchases office. Quotations delayed by mail shall not be considered, shall not be opened, and shall be rejected. Arrangements may be made for their return at the bidder's request and expense. Quotations may be faxed to the Division of Purchases and accepted if the signed quotation form and required information is faxed and received prior to the due date and time. Quotations sent by email will not be accepted.</p>
            </div>

            <div class="section">
                <h4>03. Sealed and Marked</h4>
                <p>If sent by mail, one original signed Quotation shall be submitted in one sealed package, clearly marked on the outside of the package with the Request for Quotation number and addressed to:</p>
                <div style="display: block" class="contact-info">
                    {{progress["Client Information"]["company name"]}}<br>
                    Division of Purchases<br>
                    {{''.join(progress["Client Information"]["Address"].split(',')[:-2])}}<br>
                    {{''.join(progress["Client Information"]["Address"].split(',')[-2:])}}
                </div>
            </div>

            <div class="section">
                <h4>04. Legal Name and Signature</h4>
                <p>Quotations shall clearly indicate the legal name, address, and telephone number of the bidder (company, firm, corporation, partnership, or individual). Quotations shall be manually signed above the printed name and title of signer on the Affidavit of Compliance page. The signer shall have the authority to bind the company to the submitted Quotation. Failure to properly sign the Quote form shall invalidate same, and it shall not be considered for award.</p>
            </div>
        </div>

        <div class="page-break">
            <div class="section-title">GENERAL TERMS AND CONDITIONS</div>
            
            <div class="section">
                <h4>1. PURPOSE:</h4>
                <p>These specifications establish the minimum requirements for furniture to be used {{progress["Client Information"]["Company"]}}.</p>
            </div>

            <div class="section">
                <h4>2. LOCATION:</h4>
                <p>{{progress["Client Information"]["company name"]}} {{progress["Client Information"]["Address"]}}.</p>
            </div>

            <div class="section">
                <h4>3. QUANTITIES:</h4>
                <p>Quantities listed are estimates only and may be more or less based on prices submitted and available funds.</p>
            </div>

            <div class="section">
                <h4>4. AWARD:</h4>
                <p>In making an award the City will look at price, delivery, and warranty. Delivery is desired no later than October 8, 2012 so furniture will be available for the Dan Kinney Family Center grand opening. The City reserves the right to make separate awards for each line, group, or combination thereof.</p>
            </div>

            <div class="section">
                <h4>5. SCOPE OF WORK:</h4>
                <p>a. Provide: Furniture, deliver, the Contractor shall be responsible for all installation, if required and removal and disposal of all residual packing or shipping material.</p>
            </div>

            <div class="section">
                <h4>6. NEW PRODUCT:</h4>
                <p>All products supplied hereunder shall be new and the manufacturers standard model in current production. The product shall not be rebuilt, reconditioned, or refurbished. All products supplied hereunder shall, except as specified herein, fully conform to each and every specification, drawing, sample or other description, which is furnished to the City by the manufacturer and/or the Contractor.</p>
            </div>
        </div>

        <div class="page-break">
            <div class="section-title">SPECIFICATIONS</div>
            
            <div class="section">
                {% for item in progress.get("furniture_items_and_pricing") %}
                <div class="specifications-item">
                    <h4>{{ item.reference }}</h4>
                    <p><strong>Reference:</strong> {{ item.reference }}</p>
                    <ul>
                        {% for attr in item.attributes %}
                        <li>{{ attr }}</li>
                        {% endfor %}
                        <li><strong>Quantity: {{ item.quantity }}</strong></li>
                    </ul>
                </div>
                {% endfor %}

            </div>
        </div>

        <div class="page-break">
            <div class="section-title">QUOTATION FORM - PROPOSAL</div>
            
            <div class="form-section">
                <p><strong>SUBMITTED BY:</strong> {{progress["Enterprise Information"]["name"]}} </p>
                
                <p>Pursuant to and in accordance with the above stated Request for Quotation, the undersigned hereby declares that they have examined the RFQ documents and specifications for the item(s) listed below.</p>
                
                <p>The undersigned proposes and agrees, if their Bid is accepted to furnish the item(s) submitted below, including delivery to Springfield, Missouri in accordance with the delivery schedule indicated below and according to the prices products/services information submitted.</p>
            </div>

            <table class="quotation-table">
                <thead>
                    <tr>
                        <th>ITEM</th>
                        <th>PRODUCT CODE</th>
                        <th>DESCRIPTION</th>
                        <th>QTY</th>
                        <th>UNIT PRICE</th>
                        <th>DISCOUNT PRICE</th>
                        <th>TOTAL AMOUNT</th>
                    </tr>
                </thead>
                <tbody>
                {% for item in progress.get("furniture_items_and_pricing") %}
                    <tr>
                        <td>{{ loop.index }}.</td>
                        <td>{{ item["product code"] }}
                        <td>
                            {{ item.reference }}{% if item.attributes %}, {{ item.attributes | join(', ') }}{% endif %}.<br>
                        </td>
                        <td>{{ item.quantity }}</td>
                        <td></td>
                        <td></td>
                        <td></td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>

            <div class="form-section">
                <p><strong>DELIVERY:</strong> F.O.B. DESTINATION</p>
                <p><strong>ACCEPT VISA P-CARD:</strong> YES________ NO________</p>
                <p>Prompt Payment Discount __________% _________ Days, Net _____ Days</p>
            </div>
        </div>

        <div class="page-break">
            <div class="section-title">AFFIDAVIT OF COMPLIANCE</div>
            
            <div class="form-section">
                <p><strong>To be submitted with Vendor's Quotation</strong></p>
                
                <p>_____We DO NOT take exception to the RFQ Documents/Requirements.</p>
                <p>_____We TAKE exception to the RFQ Documents/Requirements as follows:</p>
                
                <p><strong>Specific exceptions are as follows:</strong></p>
                <div style="min-height: 100px; border: 1px solid #ccc; padding: 10px; margin: 10px 0;"></div>
                
                <p>I have carefully examined the Request for Quotation and agree to abide by all submitted pricing, delivery, terms and conditions of this Quotation unless otherwise stipulated herein.</p>
                
                <table style="width: 100%; margin-top: 20px;">
                    <tr>
                        <td style="border: none; padding: 10px;">
                            <strong>Company Name:</strong> {{progress["Enterprise Information"]["name"]}}<br><br>
                            <strong>By:</strong> <span class="signature-line"></span><br>
                            <span style="margin-left: 40px;">{{progress['Enterprise Information']['contactName']}}</span><br><br>
                            
                        </td>
                        <td style="border: none; padding: 10px;">
                            <strong>Company Address:</strong><br>
                            {{progress["Enterprise Information"]["address"]}}<br><br>
                            <strong>Telephone Number:</strong> {{progress["Enterprise Information"]["phoneNumber"]}}<br><br>
                            <strong>Email:</strong> {{progress["Enterprise Information"]["email"]}}<br><br>
                            <strong>Date:</strong> {{today}}
                        </td>
                    </tr>
                </table>
                
                <div style="margin-top: 30px;">
                    <p><strong>ADDENDA</strong></p>
                    <p>Bidder acknowledges receipt of the following addendum:</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                    <p>Addendum No. ___</p>
                </div>
            </div>
        </div>


        <footer style="margin-top: 50px; text-align: center; font-size: 12px; color: #666;">
            <p>{{progress["Client Information"]["company name"]}}<br>
            {{progress["Client Information"]["Address"]}}<br>
            {% if progress["Client Information"]["phone"] %}Phone: {{progress["Client Information"]["phone"]}} {% endif %} | {% if progress["Client Information"]["fax"] %}Fax: {{progress["Client Information"]["fax"]}} {% endif %}</p>
        </footer>

    </body>
    </html>
    """

    template = Template(html_template)
    return template.render(progress=progress,today = today)