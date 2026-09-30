import os
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from config import Config

class PDFReportGenerator:
    """
    Automated ReportLab PDF Report Generator for Predictive Oncology Analytics.
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or Config.REPORTS_FOLDER
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        """Define report typography styles."""
        self.styles.add(ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1a365d'),
            alignment=1, # Centered
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            'ReportSubtitle',
            parent=self.styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#4a5568'),
            alignment=1,
            spaceAfter=15
        ))
        self.styles.add(ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#2b6cb0'),
            spaceBefore=12,
            spaceAfter=8
        ))
        self.styles.add(ParagraphStyle(
            'DisclaimerText',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#9b2c2c'),
            alignment=1
        ))

    def generate_report(
        self,
        patient_id: str,
        prediction_result: Dict[str, Any],
        dataset_info: Dict[str, Any],
        model_metrics: Dict[str, Dict[str, Any]],
        graph_paths: Dict[str, str],
        filename: Optional[str] = None
    ) -> str:
        """
        Generate complete PDF evaluation report.
        """
        if not filename:
            filename = f"Oncology_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        elements = []

        # Title Header
        elements.append(Paragraph("Predictive Oncology Analytics & QML Support", self.styles['ReportTitle']))
        elements.append(Paragraph("Rapid Identification of Drug Resistance Signatures using Quantum-Inspired Models", self.styles['ReportSubtitle']))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2b6cb0'), spaceAfter=10))

        # Disclaimer Box
        disclaimer_table = Table([[
            Paragraph(
                "RESEARCH PROTOTYPE — NOT FOR CLINICAL DIAGNOSIS OR TREATMENT DECISIONS.<br/>"
                "All probability scores and drug resistance predictions are computational model outputs for research purposes only.",
                self.styles['DisclaimerText']
            )
        ]], colWidths=[540])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff5f5')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#feb2b2')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER')
        ]))
        elements.append(disclaimer_table)
        elements.append(Spacer(1, 15))

        # 1. Prediction Summary Table
        elements.append(Paragraph("1. Patient Computational Prediction Summary", self.styles['SectionHeader']))
        
        pred_data = [
            [Paragraph("<b>Patient Identifier:</b>", self.styles['Normal']), Paragraph(patient_id, self.styles['Normal'])],
            [Paragraph("<b>Model Used:</b>", self.styles['Normal']), Paragraph(prediction_result.get('model_name', 'N/A'), self.styles['Normal'])],
            [Paragraph("<b>Computational Prediction:</b>", self.styles['Normal']), Paragraph(f"<b>{prediction_result.get('prediction', 'N/A')}</b>", self.styles['Normal'])],
            [Paragraph("<b>Resistance Probability Score:</b>", self.styles['Normal']), Paragraph(f"{prediction_result.get('probability', 0.0) * 100:.1f}%", self.styles['Normal'])],
            [Paragraph("<b>Report Generated Date:</b>", self.styles['Normal']), Paragraph(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'), self.styles['Normal'])]
        ]
        
        pred_table = Table(pred_data, colWidths=[200, 340])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#edf2f7')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(pred_table)
        elements.append(Spacer(1, 10))

        # 1b. Computational Therapeutic Regimens
        t_info = prediction_result.get('therapeutic_info', {})
        if t_info:
            elements.append(Paragraph(f"<b>Computational Therapeutic Category:</b> {t_info.get('category', 'N/A')}", self.styles['Normal']))
            elements.append(Spacer(1, 5))
            med_rows = [['Agent Name', 'Therapeutic Class', 'Biological Target']]
            for m in t_info.get('medicines', []):
                med_rows.append([m.get('name', ''), m.get('class', ''), m.get('target', '')])
            med_table = Table(med_rows, colWidths=[180, 200, 160])
            med_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            elements.append(med_table)
            elements.append(Spacer(1, 15))

        # 2. Dataset Overview
        elements.append(Paragraph("2. Dataset Benchmark Parameters", self.styles['SectionHeader']))
        ds_data = [
            ['Dataset File', 'Total Records', 'Feature Count', 'Target Column'],
            [
                str(dataset_info.get('filename', 'Demo Dataset')),
                str(dataset_info.get('rows', 'N/A')),
                str(dataset_info.get('columns', 'N/A')),
                str(dataset_info.get('target_column', 'target'))
            ]
        ]
        ds_table = Table(ds_data, colWidths=[200, 110, 110, 120])
        ds_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(ds_table)
        elements.append(Spacer(1, 15))

        # 3. Model Benchmark Comparison
        elements.append(Paragraph("3. Classical vs Quantum Model Performance Benchmark", self.styles['SectionHeader']))
        comp_headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Train Time']
        comp_rows = [comp_headers]

        for m_name, m_val in model_metrics.items():
            comp_rows.append([
                m_name,
                f"{m_val.get('accuracy', 0):.4f}",
                f"{m_val.get('precision', 0):.4f}",
                f"{m_val.get('recall', 0):.4f}",
                f"{m_val.get('f1_score', 0):.4f}",
                f"{m_val.get('roc_auc', 0):.4f}",
                f"{m_val.get('training_time', 0):.2f}s"
            ])

        comp_table = Table(comp_rows, colWidths=[120, 70, 70, 70, 70, 70, 70])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(comp_table)
        elements.append(Spacer(1, 15))

        # 4. Embedded Visualization Plots
        elements.append(Paragraph("4. Benchmark Visualizations", self.styles['SectionHeader']))
        
        img_tables = []
        roc_img = graph_paths.get('roc')
        cm_img = graph_paths.get('cm')
        fi_img = graph_paths.get('fi')

        if roc_img and os.path.exists(roc_img) and cm_img and os.path.exists(cm_img):
            img1 = Image(roc_img, width=250, height=200)
            img2 = Image(cm_img, width=250, height=200)
            elements.append(Table([[img1, img2]], colWidths=[270, 270]))
            elements.append(Spacer(1, 10))

        if fi_img and os.path.exists(fi_img):
            img3 = Image(fi_img, width=480, height=240)
            elements.append(Paragraph("<b>Top Feature Resistance Signatures</b>", self.styles['Normal']))
            elements.append(img3)

        # Build PDF Document
        doc.build(elements)
        return filepath
