/*
 * Copyright (c) 2007, Swedish Institute of Computer Science.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file is part of the Contiki operating system.
 *
 */

/**
 * \file
 *         A mesh routing protocol
 * \author
 *         Adam Dunkels <adam@sics.se>
 */

/**
 * \addtogroup rimemesh
 * @{
 */

#include "contiki.h"
#include "net/rime/rime.h"
#include "route.h"
#include "mesh.h"
#include "node-id.h"
#include "dual_conf.h"
#include <stddef.h> /* For offsetof */

#define PACKET_TIMEOUT (CLOCK_SECOND * 30)    // 30 seconds timeout 

#define DEBUG 0
#if DEBUG
#include <stdio.h>
#define PRINTF(...) printf(__VA_ARGS__)
#else
#define PRINTF(...)
#endif
static struct ctimer mrt;
extern struct unicast_conn uc;  // Unicast to controller for missroute
extern linkaddr_t br_id; // Border router ID
extern int c_send;
extern int s_send;
/*---------------------------------------------------------------------------*/
static void
data_packet_received(struct multihop_conn *multihop,
		     const linkaddr_t *from,
		     const linkaddr_t *prevhop, uint8_t hops)
{
  struct mesh_conn *c = (struct mesh_conn *)
    ((char *)multihop - offsetof(struct mesh_conn, multihop));

  struct route_entry *rt;

  /* Refresh the route when we hear a packet from a neighbor. */
  rt = route_lookup(from);
  if(rt != NULL) {
    route_refresh(rt);
  }
  
  if(c->cb->recv) {
    c->cb->recv(c, from, hops);
  }
}

/*---------------------------------------------------------------------------*/
/* PACKET TIMEOUT HANDLER  triggered by the timer */
static void
miss_route_handler(void *ptr)
{
  struct route_entry *rt;
  struct mesh_conn *c = ptr;

  if(c->queued_data != NULL ) {  
      rt = route_lookup(&c->queued_data_dest);   // Check again for route  
      queuebuf_to_packetbuf(c->queued_data);
      queuebuf_free(c->queued_data);
      c->queued_data = NULL;

    if(rt != NULL) {
      PRINTF("mesh.c Route miss: Controller returned route Resending...\n");
      multihop_resend(&c->multihop, &rt->nexthop);
      if(c->cb->sent != NULL) {
        c->cb->sent(c);
      }
    } 
    else {
      PRINTF("mesh.c Route miss: timeout, timed out\n");
      if(c->cb->timedout != NULL) {
        c->cb->timedout(c);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
void data_packet_resend(struct mesh_conn *c){
   PRINTF("Stoping timeout timer\n");
   ctimer_stop(&mrt);      // Stop the timer
   miss_route_handler(c);  // resend the packet
}

/*---------------------------------------------------------------------------*/
static linkaddr_t *
data_packet_forward(struct multihop_conn *multihop,
		    const linkaddr_t *originator,
		    const linkaddr_t *dest,
		    const linkaddr_t *prevhop, uint8_t hops)
{
   struct route_entry *rt;
   struct mesh_conn *c = (struct mesh_conn *)
    ((char *)multihop - offsetof(struct mesh_conn, multihop));

   rt = route_lookup(dest);  // Tryf ayto prepei panata na bgazei kati an bgalei NULL routeMISS
  
   if(rt == NULL) {  // If route MISS
      if(c->queued_data != NULL) {
         queuebuf_free(c->queued_data);
      }

      PRINTF("data_packet_forward: queueing data\n");
      c->queued_data = queuebuf_new_from_packetbuf();
      linkaddr_copy(&c->queued_data_dest, dest);     
      PRINTF("data_packet_forward: sending route request\n");

      ctimer_set(&mrt, PACKET_TIMEOUT, miss_route_handler, c);   // Set-start Miss Route timer

      char msgbuf[11];
      //PAYLOAD:  PTY NID  DID
      //SIZE(11):  2   4    4
      //EXAMPLE:  MR  0200 0300
      sprintf(msgbuf,"MR%02d00%02u%02u", node_id, dest->u8[0], dest->u8[1]);  //ADD ENG fix tryfon ?????????????????????
      packetbuf_copyfrom(msgbuf, sizeof(msgbuf));
           
      if(br_id.u8[0]!=node_id){ // Any Node
         dual_radio_switch(LONG_RADIO);
         printf("Sending LR unicast to %d.%d Miss Routing msg[%s]\n",br_id.u8[0],br_id.u8[1],msgbuf); 
         unicast_send(&uc,&br_id);   // SEND LR UNICAST New node RESPONCE
         c_send++;  
      }
      else { // Boder Router
         printf("{\"PTY\":\"MR\",\"NID\":\"%02d.%02d\",\"DID\":\"%02d.%02d\"}\n", br_id.u8[0], br_id.u8[1], dest->u8[0], dest->u8[1]);  
         s_send++;  
      }
      return NULL;
   } 
   else {
      route_refresh(rt);
   }
   return &rt->nexthop;  
}

/*---------------------------------------------------------------------------*/
static const struct multihop_callbacks data_callbacks = { data_packet_received,
						    data_packet_forward };
/*---------------------------------------------------------------------------*/
void
mesh_open(struct mesh_conn *c, uint16_t channels,
	  const struct mesh_callbacks *callbacks)
{
  // route_init();  // Tryf: I initialize in coral.c
  multihop_open(&c->multihop, channels, &data_callbacks);
 /* route_discovery_open(&c->route_discovery_conn,   // Tryf Apenergopoio to auto route discovery
		       CLOCK_SECOND * 2,
		       channels + 1,
		       &route_discovery_callbacks); */
  c->cb = callbacks;
}
/*---------------------------------------------------------------------------*/
void
mesh_close(struct mesh_conn *c) {
   multihop_close(&c->multihop);
   ctimer_stop(&mrt);
}
/*---------------------------------------------------------------------------*/
int
mesh_send(struct mesh_conn *c, const linkaddr_t *to)
{
  int could_send;

  PRINTF("%d.%d: mesh_send to %d.%d\n",
	 linkaddr_node_addr.u8[0], linkaddr_node_addr.u8[1],to->u8[0], to->u8[1]);

  could_send = multihop_send(&c->multihop, to);

  if(!could_send) {
    PRINTF("mesh_send: could not send\n");
    return 0;
  }

  if(c->cb->sent != NULL) {
    c->cb->sent(c);
  } 
  return 1;
}
/*---------------------------------------------------------------------------*/
int
mesh_ready(struct mesh_conn *c)
{
  return (c->queued_data == NULL);
}


/** @} */
