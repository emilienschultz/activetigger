import { FC, useState } from 'react';

import { marked } from 'marked';
import { useGetSchemeCodebook } from '../core/api';

import { CodebookManagement } from '../components/CodeBookManagement';

import MDEditor from '@uiw/react-md-editor';
import { Modal } from 'react-bootstrap';
import { FaBookOpen, FaCloudDownloadAlt } from 'react-icons/fa';
import { MdDriveFileRenameOutline } from 'react-icons/md';

interface CodebookDisplayProps {
  projectSlug: string | null;
  currentScheme: string | null;
  canEdit?: boolean;
}

export const CodebookDisplay: FC<CodebookDisplayProps> = ({
  projectSlug,
  currentScheme,
  canEdit,
}) => {
  // get codebook
  const { codebook, time, reFetchCodebook } = useGetSchemeCodebook(
    projectSlug || null,
    currentScheme || null,
  );

  // open codebook edition
  const [showCodebookModal, setShowCodebookModal] = useState(false);
  const openAsHTML = () => {
    const htmlContent = `
    <html>
      <head>
        <title>Codebook</title>
        <meta charset="UTF-8" />
        <style>
          body { font-family: sans-serif; padding: 2em; }
        </style>
      </head>
      <body>
        <div>${marked.parse(codebook || '')}</div>
      </body>
    </html>
  `;
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
  };

  const downloadMarkdown = () => {
    const blob = new Blob([codebook || ''], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'codebook.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="d-flex justify-content-center my-3">
        <div
          className="card border-primary shadow-sm w-100"
          style={{ maxWidth: '100%', minWidth: '350px', maxHeight: '50vh' }}
        >
          {/* Header fin et discret */}
          <div
            className="card-header bg-white text-primary d-flex justify-content-between align-items-center border-bottom border-primary"
            style={{
              padding: '4px 8px', // header plus fin
              fontSize: '0.9rem',
              fontWeight: 600,
            }}
          >
            {' '}
            <span className="fw-bold d-flex align-items-center">
              📘 <span className="ms-2">Guidelines</span>
            </span>
            {canEdit && (
              <div className="btn-group btn-group-sm" role="group">
                <button
                  type="button"
                  className="btn btn-light text-primary border-0"
                  onClick={() => setShowCodebookModal(true)}
                  title="Edit codebook"
                >
                  <MdDriveFileRenameOutline size={16} />
                </button>
                <button
                  className="btn btn-light text-primary border-0"
                  onClick={openAsHTML}
                  title="Open"
                >
                  <FaBookOpen size={16} />
                </button>
                <button
                  className="btn btn-light text-primary border-0"
                  onClick={downloadMarkdown}
                  title="Download"
                >
                  <FaCloudDownloadAlt size={16} />
                </button>
              </div>
            )}
          </div>

          {/* Corps du codebook avec scroll */}
          <div
            className="card-body overflow-auto"
            style={{ height: '100%', maxHeight: 'calc(50vh - 50px)' }} // -50px pour compenser le header
            data-color-mode="light"
          >
            <MDEditor.Markdown
              source={codebook}
              style={{
                backgroundColor: 'transparent',
                fontSize: '0.95rem',
                lineHeight: '1.6',
              }}
            />
          </div>
        </div>
      </div>
      <Modal
        show={showCodebookModal}
        onHide={() => setShowCodebookModal(false)}
        id="codebook-modal"
        size="xl"
      >
        <Modal.Header closeButton>
          <Modal.Title>Edit your current codebook</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <CodebookManagement
            projectName={projectSlug}
            currentScheme={currentScheme || null}
            codebook={codebook}
            time={time}
            reFetchCodebook={reFetchCodebook}
          />
        </Modal.Body>
      </Modal>
    </div>
  );
};
