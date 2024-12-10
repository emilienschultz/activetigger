import { FC, useState } from 'react';
import { Button, Modal } from 'react-bootstrap';
import { MdRunningWithErrors } from 'react-icons/md';

export const ModalErrors: FC<{
  errors: string | string[];
}> = ({ errors }) => {
  const [show, setShow] = useState(false);

  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  const content = typeof errors === 'string' ? [errors] : errors;

  return (
    <>
      <div onClick={handleShow} className="badge text-bg-danger" style={{ cursor: 'pointer' }}>
        <MdRunningWithErrors /> Errors
      </div>

      <Modal show={show} onHide={handleClose}>
        <Modal.Header closeButton>
          <Modal.Title>Errors</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          {(content as string[]).map((e, i) => (
            <p key={i}>{e}</p>
          ))}
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={handleClose}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  );
};
