import { FC } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';

import { LoginParams, login } from '../../core/api';

export const LoginForm: FC = () => {
  const { handleSubmit, register } = useForm<LoginParams>();

  const onSubmit: SubmitHandler<LoginParams> = async (data) => {
    const response = await login(data);

    console.log(response);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input type="text" {...register('username')} />
      <input type="password" {...register('password')} />
      <button>login</button>
    </form>
  );
};
